"""Her kullanıcının en son "Günlük otomatik" yedeğini Google Drive'a gönderir.

GitHub Actions'ta (drive-yedek.yml) çalışır; uygulamanın açılışına karışmaz.
Ortam değişkenleri: DATABASE_URL (Neon), DRIVE_YEDEK_URL (Apps Script web uygulaması
adresi), DRIVE_YEDEK_ANAHTAR (Apps Script'teki gizli anahtar).
Veritabanına yalnız okuma sorgusu atar.
"""
import hashlib
import json
import os
import sys
import urllib.request

REASON = "Günlük otomatik"


def build_payloads(rows, anahtar):
    """rows: (owner, created_tr, data) satırları -> Apps Script'e gidecek gövdeler.

    Dosya adı: nobet-yedek-<kullanıcı>-<tarih>.json. Kullanıcı adı e-posta olabilir;
    dosya adında yalnız kısa bir özeti (ya da boşsa "ana") görünür.
    """
    payloads = []
    for owner, created_tr, data in rows:
        parsed = json.loads(data)  # bozuk yedek Drive'a gitmesin
        if not parsed.get("schedules") and not parsed.get("settings"):
            raise ValueError(f"Yedek boş görünüyor ({created_tr})")
        who = hashlib.sha1(owner.encode("utf-8")).hexdigest()[:6] if owner else "ana"
        day = created_tr[:10]
        payloads.append({
            "key": anahtar,
            "prefix": f"nobet-yedek-{who}-",
            "name": f"nobet-yedek-{who}-{day}.json",
            "data": data,
        })
    return payloads


def latest_rows(conn):
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT ON (owner) owner, created_tr, data
            FROM backup_snapshots WHERE reason = %s
            ORDER BY owner, id DESC
            """,
            (REASON,),
        )
        return cur.fetchall()


def send(url, payload):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "text/plain;charset=utf-8"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        answer = json.loads(resp.read().decode("utf-8"))
    if not answer.get("ok"):
        raise RuntimeError(f"Drive aracısı reddetti: {answer.get('error')}")
    return answer


def main():
    import psycopg2
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    try:
        rows = latest_rows(conn)
    finally:
        conn.close()
    if not rows:
        print("Günlük otomatik yedek bulunamadı.", file=sys.stderr)
        return 1
    for payload in build_payloads(rows, os.environ["DRIVE_YEDEK_ANAHTAR"]):
        send(os.environ["DRIVE_YEDEK_URL"], payload)
        print("Gönderildi:", payload["name"], f"({len(payload['data'])} bayt)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
