# -*- coding: utf-8 -*-
"""Diskte değişen çekirdek modülleri yeniden yükler (nobet.py her çalışmada çağırır).

Streamlit Cloud yeni kodu sunucuyu yeniden başlatmadan çekince eski modül
hafızada kalabiliyordu. Bu dosya kendisi yeniden yüklenmez: kilit sunucu
boyunca tek kalmalı. Bu dosya değiştirilirse sunucu yeniden başlatılmalıdır
(Streamlit'in dosya izleyicisi onu ``sys.modules``'tan silerse ikinci bir
kilit oluşur; yalnızca bu dosyanın kendisi değiştiğinde olabilir).

Bilinen sınır: kilit yalnızca tazeleme adımını korur. O sırada betiğin
ortasında olan başka bir oturum yeniden yüklenen modülün değerlerini görebilir.
"""
import hashlib
import importlib
import os
import sys
import threading
import traceback

STAMP = "_nobet_loaded_stamp"
_APP_DIR = os.path.dirname(os.path.abspath(__file__))

# Aynı anda iki oturum aynı modülü yeniden yüklemesin; yükleme sürerken başka
# oturum damgayı "güncel" görüp yarım modülle devam etmesin.
_LOCK = threading.RLock()


def fresh_modules(names, force=False):
    """``names`` sırasıyla dosyası değişmiş modülleri kilit altında yeniden yükler.

    Bir modül yeniden yüklenince ardından gelenler de (ona bağlı oldukları için)
    yeniden yüklenir. Damga yalnızca yükleme başarıyla bitince yazılır: yükleme
    yarıda hata verirse damga boş kalır, hata günlüğe sürümle yazılır ve sonraki
    çalışma yeniden dener. Yeniden yüklenen modül adlarını döndürür.
    """
    reloaded = []
    with _LOCK:
        for name in names:
            module = sys.modules.get(name)
            if module is None:
                continue
            try:
                stamp = os.stat(module.__file__).st_mtime_ns
            except (OSError, TypeError):
                continue
            if force or reloaded or getattr(module, STAMP, None) != stamp:
                setattr(module, STAMP, None)
                try:
                    importlib.reload(module)
                except BaseException as exc:
                    # Modül kendi damgasını yazmış olsa bile (eski dosya düzeni)
                    # boşaltılır; yarım modül "güncel" sayılmaz.
                    setattr(module, STAMP, None)
                    _log(f"{name} yeniden yüklenemedi (sürüm {code_version()}):\n"
                         + "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
                    raise
                setattr(module, STAMP, stamp)
                reloaded.append(name)
        # Sunucu açılınca ve her yeni kodda bir kez: günlükte "canlıda hangi kod var" görünsün.
        version = code_version()
        if reloaded or _version_cache.get("announced") != version:
            _version_cache["announced"] = version
            _log(f"Çalışan kod sürümü {version}"
                 + (f"; yeniden yüklendi: {', '.join(reloaded)}" if reloaded else ""))
    return reloaded


# --- Çalışan kodun sürüm etiketi ------------------------------------------------

_version_cache = {}


def _app_sources():
    """Uygulamanın (test olmayan) .py dosyaları, ada göre sıralı."""
    try:
        names = os.listdir(_APP_DIR)
    except OSError:
        return []
    return sorted(
        name for name in names
        if name.endswith(".py") and not name.startswith("test_")
    )


def _content_digest():
    """Uygulama dosyalarının içeriğinden kısa özet (dosyalar değişmedikçe önbellekten)."""
    files = _app_sources()
    key = []
    for name in files:
        try:
            info = os.stat(os.path.join(_APP_DIR, name))
        except OSError:
            continue
        key.append((name, info.st_mtime_ns, info.st_size))
    key = tuple(key)
    cached = _version_cache.get("digest")
    if cached and cached[0] == key:
        return cached[1]
    digest = hashlib.md5()
    for name, _mtime, _size in key:
        try:
            with open(os.path.join(_APP_DIR, name), "rb") as handle:
                digest.update(name.encode() + b"\0" + handle.read() + b"\0")
        except OSError:
            continue
    value = digest.hexdigest()[:7]
    _version_cache["digest"] = (key, value)
    return value


def _git_dirs():
    """(HEAD'in olduğu klasör, ref'lerin olduğu klasör); ``.git`` dosyaysa (worktree) izlenir."""
    git_dir = os.path.join(_APP_DIR, ".git")
    if os.path.isfile(git_dir):
        with open(git_dir, encoding="utf-8") as handle:
            line = handle.read().strip()
        if not line.startswith("gitdir:"):
            return None, None
        git_dir = os.path.join(_APP_DIR, line[7:].strip())
    common = git_dir
    try:
        with open(os.path.join(git_dir, "commondir"), encoding="utf-8") as handle:
            common = os.path.join(git_dir, handle.read().strip())
    except OSError:
        pass
    return git_dir, common


def _git_commit():
    """Klasördeki git deposunun kısa commit'i; okunamazsa boş (komut çalıştırmaz)."""
    try:
        git_dir, common = _git_dirs()
        if git_dir is None:
            return ""
        with open(os.path.join(git_dir, "HEAD"), encoding="utf-8") as handle:
            head = handle.read().strip()
        if not head.startswith("ref:"):
            return head[:7]
        ref = head[4:].strip()
        try:
            with open(os.path.join(common, *ref.split("/")), encoding="utf-8") as handle:
                return handle.read().strip()[:7]
        except OSError:
            with open(os.path.join(common, "packed-refs"), encoding="utf-8") as handle:
                for line in handle:
                    parts = line.split()
                    if len(parts) == 2 and parts[1] == ref:
                        return parts[0][:7]
    except (OSError, UnicodeDecodeError):
        pass
    return ""


def code_version():
    """Ör. ``"b141db1 · 3f9a2c1"``: commit (varsa) ve dosya içeriklerinin özeti.

    Commit, Cloud'un çektiği kodu; özet, diskteki dosyaların gerçekten ne olduğunu
    gösterir (commit okunamasa ya da elle değiştirilmiş dosya olsa da doğru kalır).
    """
    commit = _git_commit()
    digest = _content_digest()
    return f"{commit} · {digest}" if commit else digest


def _log(text):
    print(f"[nobet] {text}", flush=True)
