"""Replit Secrets'tan .streamlit/secrets.toml dosyasını üretir.

Streamlit giriş ayarlarını yalnızca secrets.toml'dan okur; Replit ise
gizli bilgileri ortam değişkeni olarak verir. Uygulama başlamadan önce bu
betik çalışır ve Google ile giriş için gereken [auth] bölümünü yazar.

Gerekli Replit Secrets:
  GOOGLE_CLIENT_ID      Google Cloud OAuth istemci kimliği
  GOOGLE_CLIENT_SECRET  Google Cloud OAuth istemci gizli anahtarı
  AUTH_REDIRECT_URI     https://<uygulama-adresi>/oauth2callback
  AUTH_COOKIE_SECRET    Uzun rastgele bir metin (oturum çerezini imzalar)
İsteğe bağlı:
  ADMIN_EMAIL           Girişten önce kaydedilmiş veriyi alacak Google hesabı

GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET / AUTH_REDIRECT_URI yoksa dosya
yazılmaz ve uygulama girişsiz (tek kullanıcılı) çalışır.
"""
import json
import os
import secrets
from pathlib import Path

MARKER = "# Bu dosya scripts/write_streamlit_secrets.py tarafından üretildi.\n"
SECRETS_PATH = Path(__file__).resolve().parent.parent / ".streamlit" / "secrets.toml"


def build_secrets_toml(env):
    client_id = env.get("GOOGLE_CLIENT_ID", "").strip()
    client_secret = env.get("GOOGLE_CLIENT_SECRET", "").strip()
    redirect_uri = env.get("AUTH_REDIRECT_URI", "").strip()
    if not (client_id and client_secret and redirect_uri):
        return None
    cookie_secret = env.get("AUTH_COOKIE_SECRET", "").strip()
    if not cookie_secret:
        # Yeniden başlatmada herkesin oturumu kapanır; kalıcı olması için
        # AUTH_COOKIE_SECRET tanımlayın.
        print("UYARI: AUTH_COOKIE_SECRET tanımlı değil, geçici anahtar kullanılıyor.")
        cookie_secret = secrets.token_urlsafe(48)
    values = {
        "redirect_uri": redirect_uri,
        "cookie_secret": cookie_secret,
        "client_id": client_id,
        "client_secret": client_secret,
        "server_metadata_url": "https://accounts.google.com/.well-known/openid-configuration",
    }
    # JSON dizeleri geçerli TOML temel dizeleridir.
    lines = [MARKER, "[auth]\n"]
    lines += [f"{key} = {json.dumps(value)}\n" for key, value in values.items()]
    return "".join(lines)


def main():
    content = build_secrets_toml(os.environ)
    if content is None:
        # Daha önce bu betiğin yazdığı dosya kaldıysa sil; elle yazılana dokunma.
        if SECRETS_PATH.exists() and SECRETS_PATH.read_text(encoding="utf-8").startswith(MARKER):
            SECRETS_PATH.unlink()
        print("Google girişi ayarlı değil; uygulama girişsiz çalışacak.")
        return
    if SECRETS_PATH.exists() and not SECRETS_PATH.read_text(encoding="utf-8").startswith(MARKER):
        print(f"{SECRETS_PATH} elle yazılmış; üzerine yazılmadı.")
        return
    SECRETS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SECRETS_PATH.write_text(content, encoding="utf-8")
    os.chmod(SECRETS_PATH, 0o600)
    print("Google girişi etkin.")


if __name__ == "__main__":
    main()
