# -*- coding: utf-8 -*-
"""Modül tazeleme (Mimar raporu 20 Ekim 2026, madde 4): davranış testleri.

Streamlit Cloud yeni kodu çekince nobet.py her çalışmada değişen çekirdek
modülleri yeniden yükler (nobet_tazele.py). Burada geçici "oyuncak" modüller
ve gerçek modül dosyalarının kopyalarıyla denenir; zamanlama threading.Event
ile kurulur, beklemeye (sleep) dayanmaz.
"""
import ast
import importlib
import os
import sys
import threading
import time
import types
from pathlib import Path

import pytest

import nobet_tazele

HERE = Path(__file__).parent
TAIL = "\n_nobet_loaded_stamp = os.stat(__file__).st_mtime_ns\n"


def _toy(tmp_path, monkeypatch, name, stamp_first=False):
    """``write(body)`` ile yeniden yazılan geçici modül; her yazımda dosya zamanı ileri alınır."""
    path = tmp_path / f"{name}.py"
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, name, raising=False)
    tick = [time.time_ns()]

    def write(body):
        if stamp_first:   # eski düzen: damga dosyanın başında
            path.write_text("import os\n" + TAIL + body)
        else:
            path.write_text("import os\n" + body + TAIL)
        tick[0] += 10 ** 9
        os.utime(path, ns=(tick[0], tick[0]))
    return write, path


@pytest.mark.parametrize("stamp_first", [False, True])
def test_yarida_kalan_yukleme_sonraki_calismada_yeniden_denenir(tmp_path, monkeypatch, stamp_first):
    name = f"oyuncak_yarim_{int(stamp_first)}"
    write, _ = _toy(tmp_path, monkeypatch, name, stamp_first)
    write("def surum():\n    return 'eski'\n")
    module = importlib.import_module(name)
    # Yeni kod yüklenirken ortada hata verir (ör. eksik bağımlılık).
    write("def surum():\n    return 'yeni'\nraise RuntimeError('ara hata')\ndef yeni_fonk():\n    return 1\n")
    with pytest.raises(RuntimeError):
        nobet_tazele.fresh_modules([name])
    assert module.surum() == "yeni" and not hasattr(module, "yeni_fonk")   # yarım modül
    # Sonraki çalışma (aynı dosya, düzeltilmeden) yeniden dener, damgaya kanmaz.
    with pytest.raises(RuntimeError):
        nobet_tazele.fresh_modules([name])
    write("def surum():\n    return 'yeni'\ndef yeni_fonk():\n    return 1\n")
    assert nobet_tazele.fresh_modules([name]) == [name]
    assert module.surum() == "yeni" and module.yeni_fonk() == 1
    # Değişmeyen modül yeniden yüklenmez.
    assert nobet_tazele.fresh_modules([name]) == []


@pytest.mark.parametrize("name", ["db", "nobet_core", "nobet_iyilestir"])
def test_gercek_modul_dosyasi_yarida_kalinca_yeniden_denenir(tmp_path, monkeypatch, name):
    # Gerçek dosyanın kopyası; ilk fonksiyondan hemen önce hata eklenerek yarım yükleme.
    source = (HERE / f"{name}.py").read_text(encoding="utf-8")
    copy_name = f"kopya_{name}"
    path = tmp_path / f"{copy_name}.py"
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, copy_name, raising=False)
    tick = [time.time_ns()]

    def write(text):
        path.write_text(text, encoding="utf-8")
        tick[0] += 10 ** 9
        os.utime(path, ns=(tick[0], tick[0]))

    write(source)
    module = importlib.import_module(copy_name)
    lines = source.splitlines(keepends=True)
    first_def = next(node.lineno for node in ast.parse(source).body if isinstance(node, ast.FunctionDef))
    write("".join(lines[:first_def - 1]) + "raise RuntimeError('yarım yükleme')\n" + "".join(lines[first_def - 1:]))
    with pytest.raises(RuntimeError):
        nobet_tazele.fresh_modules([copy_name])
    assert getattr(module, nobet_tazele.STAMP) != os.stat(path).st_mtime_ns
    write(source)
    assert nobet_tazele.fresh_modules([copy_name]) == [copy_name]
    assert getattr(module, nobet_tazele.STAMP) == os.stat(path).st_mtime_ns


def test_baska_oturum_yuklemenin_bitmesini_bekler(tmp_path, monkeypatch):
    # Oyuncak modül yüklenirken "basladi" der ve "devam" gelene kadar bekler.
    gate = types.ModuleType("oyuncak_kapi")
    gate.basladi, gate.devam = threading.Event(), threading.Event()
    monkeypatch.setitem(sys.modules, "oyuncak_kapi", gate)
    write, _ = _toy(tmp_path, monkeypatch, "oyuncak_kilit")
    write("def surum():\n    return 'v1'\n")
    module = importlib.import_module("oyuncak_kilit")
    write("import oyuncak_kapi\noyuncak_kapi.basladi.set()\noyuncak_kapi.devam.wait(30)\n"
          "def surum():\n    return 'v2'\ndef yeni_fonk():\n    return 2\n")

    # B oturumu kilide vardığını (ya da beklemeden bittiğini) bildirir; böylece
    # A'yı ancak B gerçekten kilitte beklerken (ya da işini bitirmişken) bırakırız.
    b_vardi = threading.Event()
    inner = nobet_tazele._LOCK

    class Lock:
        def __enter__(self):
            if threading.current_thread().name == "B":
                b_vardi.set()
            return inner.__enter__()

        def __exit__(self, *exc):
            return inner.__exit__(*exc)

    monkeypatch.setattr(nobet_tazele, "_LOCK", Lock())
    seen = {}

    def session_b():
        try:
            nobet_tazele.fresh_modules(["oyuncak_kilit"])
            seen["surum"], seen["yeni_fonk"] = module.surum(), hasattr(module, "yeni_fonk")
        finally:
            b_vardi.set()

    a = threading.Thread(target=nobet_tazele.fresh_modules, args=(["oyuncak_kilit"],), name="A")
    a.start()
    assert gate.basladi.wait(30)          # A yüklemenin ortasında
    b = threading.Thread(target=session_b, name="B")
    b.start()
    assert b_vardi.wait(30)
    gate.devam.set()
    a.join(30)
    b.join(30)
    assert seen == {"surum": "v2", "yeni_fonk": True}


def test_onceki_modul_yenilenince_sonrakiler_de_yenilenir(tmp_path, monkeypatch):
    first, _ = _toy(tmp_path, monkeypatch, "oyuncak_bir")
    second, _ = _toy(tmp_path, monkeypatch, "oyuncak_iki")
    first("X = 1\n")
    second("from oyuncak_bir import X\n")
    importlib.import_module("oyuncak_bir")
    two = importlib.import_module("oyuncak_iki")
    first("X = 2\n")
    assert nobet_tazele.fresh_modules(["oyuncak_bir", "oyuncak_iki"]) == ["oyuncak_bir", "oyuncak_iki"]
    assert two.X == 2


def test_cekirdek_moduller_damgayi_en_sonda_yazar():
    # Yükleme yarıda kalırsa modülün kendi damgası da eski kalsın.
    for name in ("db.py", "nobet_core.py", "nobet_iyilestir.py"):
        last = ast.parse((HERE / name).read_text(encoding="utf-8")).body[-1]
        assert isinstance(last, ast.Assign) and last.targets[0].id == nobet_tazele.STAMP, name


def test_yukleme_hatasi_gunluge_surumle_yazilir(tmp_path, monkeypatch, capsys):
    write, _ = _toy(tmp_path, monkeypatch, "oyuncak_gunluk")
    write("X = 1\n")
    importlib.import_module("oyuncak_gunluk")
    write("raise ValueError('bozuk kod')\n")
    with pytest.raises(ValueError):
        nobet_tazele.fresh_modules(["oyuncak_gunluk"])
    out = capsys.readouterr().out
    assert "oyuncak_gunluk yeniden yüklenemedi" in out
    assert f"sürüm {nobet_tazele.code_version()}" in out
    assert "Traceback" in out and "bozuk kod" in out


# --- Sürüm etiketi ------------------------------------------------------------

def _fake_app(tmp_path, monkeypatch):
    (tmp_path / "nobet.py").write_text("A = 1\n")
    (tmp_path / "test_x.py").write_text("B = 1\n")
    monkeypatch.setattr(nobet_tazele, "_APP_DIR", str(tmp_path))
    monkeypatch.setattr(nobet_tazele, "_version_cache", {})


def test_surum_etiketi_commit_ve_icerik_ozeti(tmp_path, monkeypatch):
    _fake_app(tmp_path, monkeypatch)
    digest = nobet_tazele.code_version()
    assert len(digest) == 7 and "·" not in digest           # git yok: yalnız özet
    # Test dosyası değişince sürüm değişmez; uygulama dosyası değişince değişir.
    (tmp_path / "test_x.py").write_text("B = 2\n")
    assert nobet_tazele.code_version() == digest
    (tmp_path / "nobet.py").write_text("A = 22\n")
    changed = nobet_tazele.code_version()
    assert changed != digest

    git = tmp_path / ".git"
    (git / "refs" / "heads").mkdir(parents=True)
    (git / "HEAD").write_text("ref: refs/heads/main\n")
    (git / "refs" / "heads" / "main").write_text("b141db1aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n")
    assert nobet_tazele.code_version() == f"b141db1 · {changed}"
    # Sıkıştırılmış ref (packed-refs) ve ayrık HEAD.
    (git / "refs" / "heads" / "main").unlink()
    (git / "packed-refs").write_text("# pack-refs\ne4b8c77bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb refs/heads/main\n")
    assert nobet_tazele.code_version().startswith("e4b8c77 · ")
    (git / "HEAD").write_text("9d83aaccccccccccccccccccccccccccccccccccc\n")
    assert nobet_tazele.code_version().startswith("9d83aac · ")
    # git worktree: .git bir dosya, ref'ler ortak klasörde.
    import shutil
    shutil.move(str(git), str(tmp_path / "ana_git"))
    worktree = tmp_path / "ana_git" / "worktrees" / "wt"
    worktree.mkdir(parents=True)
    (worktree / "commondir").write_text("../..\n")
    (worktree / "HEAD").write_text("ref: refs/heads/main\n")
    (git.parent / ".git").write_text(f"gitdir: {worktree}\n")
    assert nobet_tazele.code_version().startswith("e4b8c77 · ")


def test_surum_degisince_gunluge_bir_kez_yazilir(tmp_path, monkeypatch, capsys):
    _fake_app(tmp_path, monkeypatch)
    nobet_tazele.fresh_modules([])
    nobet_tazele.fresh_modules([])
    assert capsys.readouterr().out.count("Çalışan kod sürümü") == 1
    (tmp_path / "nobet.py").write_text("A = 3\n")
    nobet_tazele.fresh_modules([])
    assert f"Çalışan kod sürümü {nobet_tazele.code_version()}" in capsys.readouterr().out


def test_beklenmeyen_hata_gunlugunde_surum_var(capsys):
    import db
    try:
        raise TypeError("kod hatası")
    except TypeError as exc:
        db.log_unexpected("Deneme", exc)
    out = capsys.readouterr().out
    assert f"sürüm {nobet_tazele.code_version()}" in out and "kod hatası" in out


# --- Sayfa altındaki küçük sürüm satırı ----------------------------------------

def _version_lines(at):
    return [m.value for m in at.markdown if "surum-etiketi" in m.value]


def test_sayfa_altinda_kucuk_surum_satiri():
    from test_koruma_db import _app, _simulate, _stop
    at, patches = _app(lambda *a, **k: True)
    try:
        _simulate(at)
        lines = _version_lines(at)
    finally:
        _stop(patches)
    assert len(lines) == 1 and nobet_tazele.code_version() in lines[0]
    assert "font-size:0.7rem" in lines[0]
    # Sayfanın en altında.
    assert at.markdown[-1].value == lines[0]


def test_paylasim_sayfasinda_da_surum_satiri():
    from test_koruma_db import _app, _stop
    at, patches = _app(lambda *a, **k: True, team=None,
                       resolve_share_token=lambda token, raise_on_error=False: None)
    at.query_params["paylas"] = "abc123"
    try:
        at.run()
        lines = _version_lines(at)
    finally:
        _stop(patches)
    assert len(lines) == 1 and nobet_tazele.code_version() in lines[0]
