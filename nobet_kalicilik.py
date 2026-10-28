# -*- coding: utf-8 -*-
"""Bir ayın otomatik kaydının durdurulması ve yeniden açılması (tek yer).

Eskiden bu karar nobet.py'de dokuz ayrı yerde, nedeni saklanmadan veriliyordu
(kilit düğmesinin bozuk ayın durdurmasını kaldırması bundan çıkmıştı). Burada
her durdurma nedeniyle yazılır ve yalnız aynı nedenle kaldırılır.

Streamlit'e bağlı değil: ``ss`` session_state ya da düz bir sözlüktür.
Anahtar adları eskisiyle aynı (testler ve nobet_talepler.py bunları okur).

Nedenler: "okuma" (kayıt okunamadı), "cevrimdisi" (veritabanı yok),
"bozuk" (okunamayan bölüm), "kilitli" (ay kilitli), "cakisma" (başka yerde
değişmiş), "surum_bilinmiyor" (kayıtlı sürüm okunamadı).
"""
import os

# Ay değiştirilirken bir kez uyarılan nedenler ve uyarı türü (period_revert_reason).
_UYARILI = {"cakisma": "conflict", "surum_bilinmiyor": "version_unknown"}


def kaydi_durdur(ss, donem, neden):
    donem = tuple(donem)
    ss["autosave_blocked_period"] = donem
    ss["autosave_block_reason"] = neden
    if neden in _UYARILI:
        ss["autosave_conflict_period"] = donem
        ss["autosave_conflict_kind"] = _UYARILI[neden]


def kayit_durdu_mu(ss, donem):
    return ss.get("autosave_blocked_period") == tuple(donem)


def engeli_kaldir(ss, donem, neden):
    """Bu ayın kaydı ``neden`` ile durdurulduysa yeniden açar; açıldıysa True."""
    if not kayit_durdu_mu(ss, donem) or ss.get("autosave_block_reason") != neden:
        return False
    ss["autosave_blocked_period"] = None
    ss["autosave_block_reason"] = None
    return True


def engelleri_sifirla(ss):
    """Yeni ay yüklenirken önceki ayın durdurma bilgisi temizlenir."""
    for key in ("autosave_blocked_period", "autosave_block_reason",
                "autosave_conflict_period", "autosave_conflict_kind", "conflict_leave_ack"):
        ss[key] = None


# Damga en sonda: yükleme yarıda kalırsa sonraki çalışma yeniden dener (nobet_tazele.py).
_nobet_loaded_stamp = os.stat(__file__).st_mtime_ns
