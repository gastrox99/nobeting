import importlib.util
import json
import pathlib

import pytest

_spec = importlib.util.spec_from_file_location(
    "drive_yedek", pathlib.Path(__file__).parent / ".github" / "scripts" / "drive_yedek.py")
dy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dy)

GOOD = json.dumps({"schedules": [{"year": 2026, "month": 11}], "settings": {}})


def test_payload_dosya_adi_ve_icerik():
    (p,) = dy.build_payloads([("a@b.com", "2026-11-17 08:00", GOOD)], "gizli")
    assert p["name"].startswith("nobet-yedek-") and p["name"].endswith("-2026-11-17.json")
    assert "a@b.com" not in p["name"] and p["key"] == "gizli" and p["data"] == GOOD
    assert p["name"].startswith(p["prefix"])


def test_bos_owner_ana():
    (p,) = dy.build_payloads([("", "2026-11-17 08:00", GOOD)], "k")
    assert p["name"] == "nobet-yedek-ana-2026-11-17.json"


def test_bozuk_veya_bos_yedek_gonderilmez():
    with pytest.raises(ValueError):
        dy.build_payloads([("", "2026-11-17 08:00", "{}")], "k")
    with pytest.raises(json.JSONDecodeError):
        dy.build_payloads([("", "2026-11-17 08:00", "bozuk")], "k")


def test_send_reddi_hata_verir(monkeypatch):
    class R:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def read(self): return b'{"ok": false, "error": "anahtar"}'
    monkeypatch.setattr(dy.urllib.request, "urlopen", lambda *a, **k: R())
    with pytest.raises(RuntimeError):
        dy.send("http://x", {"a": 1})
