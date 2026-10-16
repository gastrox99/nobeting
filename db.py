# -*- coding: utf-8 -*-
import os
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
import pandas as pd
from io import StringIO
import json
import re

_last_error = None


def _database_url():
    """Read the URL at connect time, not at import time.

    On Streamlit Cloud the value comes from the app's Secrets; reading it on
    every connect means a newly added DATABASE_URL works without a stale
    module-level copy.
    """
    url = os.getenv('DATABASE_URL')
    if url:
        return url
    try:
        import streamlit as st
        url = st.secrets.get('DATABASE_URL')
    except Exception:
        url = None
    return url or None


def is_database_configured():
    """True when a DATABASE_URL is set (reachable or not)."""
    return bool(_database_url())


def _connect():
    return psycopg2.connect(_database_url(), connect_timeout=10)


def _describe_error(exc):
    """One-line, password-free reason for the UI."""
    message = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
    message = re.sub(r'(://[^:/@\s]+):[^@\s]+@', r'\1:***@', message)
    return message[:200]


def _log_error(action, exc):
    """Sunucu günlüğüne şifresiz hata satırı yazar."""
    print(f"{action} error: {_describe_error(exc)}")


def get_last_error():
    """Why the last init_db() failed, or None."""
    return _last_error


def init_db():
    """Initialize database schema"""
    global _last_error
    conn = None
    if not _database_url():
        _last_error = "DATABASE_URL tanımlı değil (Streamlit Secrets'a ekleyin)"
        return False
    try:
        conn = _connect()
        with conn:  # auto-commit on success, rollback on exception
            with conn.cursor() as cur:
                # Create schedules table
                cur.execute('''
                    CREATE TABLE IF NOT EXISTS schedules (
                        id SERIAL PRIMARY KEY,
                        name VARCHAR(255) NOT NULL,
                        year INT NOT NULL,
                        month INT NOT NULL,
                        team_members TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(name, year, month)
                    )
                ''')

                # Create schedule_data table
                cur.execute('''
                    CREATE TABLE IF NOT EXISTS schedule_data (
                        id SERIAL PRIMARY KEY,
                        schedule_id INT NOT NULL REFERENCES schedules(id) ON DELETE CASCADE,
                        person VARCHAR(255) NOT NULL,
                        day_col VARCHAR(50) NOT NULL,
                        assigned BOOLEAN DEFAULT FALSE,
                        UNIQUE(schedule_id, person, day_col)
                    )
                ''')

                # Add pref_json column if it doesn't exist (safe migration)
                cur.execute('''
                    ALTER TABLE schedules ADD COLUMN IF NOT EXISTS pref_json TEXT
                ''')

                # Add settings_json column if it doesn't exist (safe migration)
                cur.execute('''
                    ALTER TABLE schedules ADD COLUMN IF NOT EXISTS settings_json TEXT
                ''')

                # Add role assignment rows if they don't exist (safe migration)
                cur.execute('''
                    ALTER TABLE schedules ADD COLUMN IF NOT EXISTS rows_liste_json TEXT
                ''')

                # Team list and settings, so a page reload restores them.
                cur.execute('''
                    CREATE TABLE IF NOT EXISTS app_settings (
                        key VARCHAR(320) PRIMARY KEY,
                        value TEXT NOT NULL,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                cur.execute('''
                    ALTER TABLE app_settings ALTER COLUMN key TYPE VARCHAR(320)
                ''')

                # Per-user data (Google login): each schedule belongs to an
                # owner e-mail. '' is the single-user / pre-login data.
                cur.execute('''
                    ALTER TABLE schedules ADD COLUMN IF NOT EXISTS owner VARCHAR(320) NOT NULL DEFAULT ''
                ''')
                cur.execute('''
                    ALTER TABLE schedules DROP CONSTRAINT IF EXISTS schedules_name_year_month_key
                ''')
                cur.execute('''
                    CREATE UNIQUE INDEX IF NOT EXISTS schedules_owner_name_year_month_key
                    ON schedules (owner, name, year, month)
                ''')

                # Read-only share links: one per owner and month.
                cur.execute('''
                    CREATE TABLE IF NOT EXISTS share_links (
                        token VARCHAR(64) PRIMARY KEY,
                        owner VARCHAR(320) NOT NULL DEFAULT '',
                        year INT NOT NULL,
                        month INT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE (owner, year, month)
                    )
                ''')
        _last_error = None
        return True
    except Exception as e:
        _log_error("Database init", e)
        _last_error = _describe_error(e)
        return False
    finally:
        if conn is not None:
            conn.close()

def settings_key(owner=""):
    """app_settings row key for a user ('' = single-user mode)."""
    return f"user:{owner}" if owner else "default"


def save_schedule(name, year, month, team_members, schedule_df, pref_df=None, settings_dict=None, rows_liste=None, owner=""):
    """Save a schedule to database"""
    conn = None
    try:
        conn = _connect()

        # JSON-safe team encoding (handles commas in names)
        team_str = json.dumps(list(team_members))

        # Serialize preference dataframe as JSON if provided
        pref_json = None
        if pref_df is not None:
            try:
                pref_json = pref_df.to_json()
            except Exception:
                pref_json = None

        # Serialize settings dict as JSON if provided
        settings_json_str = None
        if settings_dict is not None:
            try:
                settings_json_str = json.dumps(settings_dict)
            except Exception:
                settings_json_str = None

        # Serialize the displayed role assignments so manual swaps survive reloads.
        rows_liste_json = None
        if rows_liste is not None:
            try:
                rows_liste_json = json.dumps(rows_liste, ensure_ascii=False)
            except Exception:
                rows_liste_json = None

        with conn:  # auto-commit on success, rollback on exception
            with conn.cursor() as cur:
                # Insert or update schedule
                cur.execute('''
                    INSERT INTO schedules (name, year, month, team_members, pref_json, settings_json, rows_liste_json, owner, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (owner, name, year, month) DO UPDATE
                        SET updated_at = CURRENT_TIMESTAMP,
                            team_members = EXCLUDED.team_members,
                            pref_json = EXCLUDED.pref_json,
                            settings_json = EXCLUDED.settings_json,
                            rows_liste_json = EXCLUDED.rows_liste_json
                    RETURNING id
                ''', (name, year, month, team_str, pref_json, settings_json_str, rows_liste_json, owner))

                schedule_id = cur.fetchone()[0]

                # Delete old data for this schedule
                cur.execute('DELETE FROM schedule_data WHERE schedule_id = %s', (schedule_id,))

                # Insert schedule data (tek toplu sorgu)
                rows = [
                    (schedule_id, person, col, bool(schedule_df.at[person, col]))
                    for person in schedule_df.index
                    for col in schedule_df.columns
                ]
                if rows:
                    execute_values(cur, '''
                        INSERT INTO schedule_data (schedule_id, person, day_col, assigned)
                        VALUES %s
                    ''', rows)

        return True
    except Exception as e:
        _log_error("Save schedule", e)
        return False
    finally:
        if conn is not None:
            conn.close()

def load_schedule(name, year, month, raise_on_error=False, owner=""):
    """Load a schedule from database.
    Returns (team_members, schedule_df, pref_df, settings_dict, rows_liste).
    pref_df, settings_dict, and rows_liste are None when not saved.

    ``raise_on_error`` is intended for interactive callers that need to
    distinguish a missing schedule from a database failure. The default keeps
    the legacy tuple-on-error behavior for non-interactive callers.
    """
    conn = None
    try:
        conn = _connect()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Get schedule
            cur.execute('''
                SELECT * FROM schedules WHERE owner = %s AND name = %s AND year = %s AND month = %s
            ''', (owner, name, year, month))

            schedule_row = cur.fetchone()
            if not schedule_row:
                return None, None, None, None, None  # finally still runs — conn will be closed

            schedule_id = schedule_row['id']

            # Decode team_members — try JSON first (new format), fall back to comma split (legacy)
            raw_team = schedule_row['team_members']
            try:
                team_members = json.loads(raw_team)
            except (json.JSONDecodeError, TypeError):
                team_members = [m.strip() for m in raw_team.split(',') if m.strip()]

            # Decode pref_df if available
            pref_df = None
            raw_pref = schedule_row.get('pref_json')
            if raw_pref:
                try:
                    # pandas 3 treats a bare string as a file path, so the
                    # saved JSON must be wrapped in a file-like object.
                    pref_df = pd.read_json(StringIO(raw_pref))
                    # "1", "2" gibi rakamdan oluşan isimler sayıya dönüşmesin.
                    pref_df.index = pref_df.index.map(str)
                    pref_df.columns = pref_df.columns.map(str)
                except Exception:
                    pref_df = None

            # Decode settings_dict if available
            settings_dict = None
            raw_settings = schedule_row.get('settings_json')
            if raw_settings:
                try:
                    settings_dict = json.loads(raw_settings)
                except Exception:
                    settings_dict = None

            # Decode role assignment rows if available. Older records do not have
            # this value, so callers can fall back to automatic role generation.
            rows_liste = None
            raw_rows_liste = schedule_row.get('rows_liste_json')
            if raw_rows_liste:
                try:
                    decoded_rows = json.loads(raw_rows_liste)
                    if isinstance(decoded_rows, list) and all(isinstance(row, dict) for row in decoded_rows):
                        rows_liste = decoded_rows
                except Exception:
                    rows_liste = None

            # Get schedule data
            cur.execute('''
                SELECT person, day_col, assigned FROM schedule_data
                WHERE schedule_id = %s ORDER BY person, day_col
            ''', (schedule_id,))

            data_rows = cur.fetchall()

        if not data_rows:
            return team_members, None, pref_df, settings_dict, rows_liste

        # Reconstruct dataframe
        schedule_dict = {}
        for row in data_rows:
            person = row['person']
            day_col = row['day_col']
            assigned = row['assigned']

            if day_col not in schedule_dict:
                schedule_dict[day_col] = {}
            schedule_dict[day_col][person] = assigned

        df = pd.DataFrame(schedule_dict, index=team_members)
        return team_members, df, pref_df, settings_dict, rows_liste

    except Exception as e:
        _log_error("Load schedule", e)
        if raise_on_error:
            raise
        return None, None, None, None, None
    finally:
        if conn is not None:
            conn.close()


def load_auto_saved_assignments(periods, owner="", include_preferences=False):
    """Otomatik kayıtlı ayların nöbetlerini tek bağlantıda okur.

    ``periods``: [(yıl, ay), ...]. Dönen değer {(yıl, ay): DataFrame};
    satırlar o ayın ekibi, sütunlar nöbet olan günler (True/False).
    Kaydı olmayan ay yer almaz. Hata olursa yukarı iletilir.
    ``include_preferences`` ise (çizelgeler, {(yıl, ay): kayıtlı tercih
    JSON'u}) döner; müsait değil günleri bundan sayılır.
    """
    wanted = {f"Otomatik_{year}_{month:02d}": (year, month) for year, month in periods}
    if not wanted:
        return ({}, {}) if include_preferences else {}
    conn = None
    try:
        conn = _connect()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute('''
                SELECT s.name, s.year, s.month, s.team_members, d.person, d.day_col
                FROM schedules AS s
                LEFT JOIN schedule_data AS d ON d.schedule_id = s.id AND d.assigned
                WHERE s.owner = %s AND s.name = ANY(%s)
            ''', (owner, list(wanted)))
            rows = cur.fetchall()
            pref_rows = []
            if include_preferences:
                # Tercih JSON'u büyük; nöbet satırlarıyla birleştirilmeden ayrıca okunur.
                cur.execute('''
                    SELECT name, year, month, pref_json FROM schedules
                    WHERE owner = %s AND name = ANY(%s)
                ''', (owner, list(wanted)))
                pref_rows = cur.fetchall()
    finally:
        if conn is not None:
            conn.close()

    preferences = {
        wanted[row['name']]: row.get('pref_json') for row in pref_rows
        if wanted.get(row['name']) == (row['year'], row['month'])
    }
    teams, cells = {}, {}
    for row in rows:
        period = wanted.get(row['name'])
        if period is None or period != (row['year'], row['month']):
            continue
        if period not in teams:
            try:
                team = json.loads(row['team_members'])
            except (json.JSONDecodeError, TypeError):
                team = [m.strip() for m in str(row['team_members'] or '').split(',') if m.strip()]
            teams[period] = [str(member) for member in team] if isinstance(team, list) else []
            cells[period] = []
        if row['person'] is not None and row['day_col'] is not None:
            cells[period].append((row['person'], row['day_col']))

    result = {}
    for period, team in teams.items():
        people = list(dict.fromkeys(team + [person for person, _ in cells[period]]))
        columns = list(dict.fromkeys(day_col for _, day_col in cells[period]))
        frame = pd.DataFrame(False, index=people, columns=columns, dtype=bool)
        for person, day_col in cells[period]:
            frame.at[person, day_col] = True
        result[period] = frame
    return (result, preferences) if include_preferences else result


def save_app_settings(settings, owner=""):
    """Save the team list and settings as one JSON document."""
    conn = None
    try:
        conn = _connect()
        with conn:
            with conn.cursor() as cur:
                cur.execute('''
                    INSERT INTO app_settings (key, value, updated_at)
                    VALUES (%s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (key) DO UPDATE
                        SET value = EXCLUDED.value,
                            updated_at = CURRENT_TIMESTAMP
                ''', (settings_key(owner), json.dumps(settings, ensure_ascii=False)))
        return True
    except Exception as e:
        _log_error("Save settings", e)
        return False
    finally:
        if conn is not None:
            conn.close()


def load_app_settings(owner="", raise_on_error=False):
    """Return saved settings as a dict, or None when nothing is saved yet."""
    conn = None
    try:
        conn = _connect()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute('SELECT value FROM app_settings WHERE key = %s', (settings_key(owner),))
            row = cur.fetchone()
        if not row:
            return None
        settings = json.loads(row['value'])
        return settings if isinstance(settings, dict) else None
    except Exception as e:
        _log_error("Load settings", e)
        if raise_on_error:
            raise
        return None
    finally:
        if conn is not None:
            conn.close()


def claim_legacy_data(owner):
    """Give data saved before login ('' owner) to this user, once.

    Only rows the user does not already have are moved, so it is safe to call
    on every login. Returns True on success.
    """
    if not owner:
        return False
    conn = None
    try:
        conn = _connect()
        with conn:
            with conn.cursor() as cur:
                cur.execute('''
                    UPDATE schedules AS legacy SET owner = %s
                    WHERE legacy.owner = ''
                      AND NOT EXISTS (
                          SELECT 1 FROM schedules AS mine
                          WHERE mine.owner = %s AND mine.name = legacy.name
                            AND mine.year = legacy.year AND mine.month = legacy.month
                      )
                ''', (owner, owner))
                cur.execute('''
                    INSERT INTO app_settings (key, value, updated_at)
                    SELECT %s, value, CURRENT_TIMESTAMP FROM app_settings WHERE key = 'default'
                    ON CONFLICT (key) DO NOTHING
                ''', (settings_key(owner),))
        return True
    except Exception as e:
        _log_error("Claim legacy data", e)
        return False
    finally:
        if conn is not None:
            conn.close()


def create_share_link(owner, year, month):
    """Create (or replace) the month's share token; the old link stops working."""
    import secrets as _secrets
    token = _secrets.token_urlsafe(16)
    conn = None
    try:
        conn = _connect()
        with conn:
            with conn.cursor() as cur:
                cur.execute('''
                    INSERT INTO share_links (token, owner, year, month)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (owner, year, month) DO UPDATE
                        SET token = EXCLUDED.token, created_at = CURRENT_TIMESTAMP
                ''', (token, owner, year, month))
        return token
    except Exception as e:
        _log_error("Create share link", e)
        return None
    finally:
        if conn is not None:
            conn.close()


def get_share_link(owner, year, month):
    """The month's current share token, or None."""
    conn = None
    try:
        conn = _connect()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                'SELECT token FROM share_links WHERE owner = %s AND year = %s AND month = %s',
                (owner, year, month),
            )
            row = cur.fetchone()
        return row['token'] if row else None
    except Exception as e:
        _log_error("Get share link", e)
        return None
    finally:
        if conn is not None:
            conn.close()


def delete_share_link(owner, year, month):
    conn = None
    try:
        conn = _connect()
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    'DELETE FROM share_links WHERE owner = %s AND year = %s AND month = %s',
                    (owner, year, month),
                )
        return True
    except Exception as e:
        _log_error("Delete share link", e)
        return False
    finally:
        if conn is not None:
            conn.close()


def resolve_share_token(token):
    """(owner, year, month) for a valid token, else None."""
    if not token or not isinstance(token, str) or len(token) > 64:
        return None
    conn = None
    try:
        conn = _connect()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute('SELECT owner, year, month FROM share_links WHERE token = %s', (token,))
            row = cur.fetchone()
        return (row['owner'], row['year'], row['month']) if row else None
    except Exception as e:
        _log_error("Resolve share link", e)
        return None
    finally:
        if conn is not None:
            conn.close()


def export_owner_data(owner=""):
    """Kullanıcının tüm çizelgelerini ve ayarlarını okur (yedek için).

    Dönen değer (ayarlar, çizelgeler). Paylaşım bağlantıları yedeğe
    girmez. Hata olursa yukarı iletilir.
    """
    conn = None
    try:
        conn = _connect()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute('SELECT value FROM app_settings WHERE key = %s', (settings_key(owner),))
            settings_row = cur.fetchone()
            cur.execute('''
                SELECT id, name, year, month, team_members, pref_json, settings_json, rows_liste_json
                FROM schedules WHERE owner = %s ORDER BY year, month, name
            ''', (owner,))
            schedules = [dict(row) for row in cur.fetchall()]
            cur.execute('''
                SELECT d.schedule_id, d.person, d.day_col, d.assigned
                FROM schedule_data AS d JOIN schedules AS s ON s.id = d.schedule_id
                WHERE s.owner = %s ORDER BY d.schedule_id, d.id
            ''', (owner,))
            cell_rows = cur.fetchall()
    finally:
        if conn is not None:
            conn.close()

    settings = None
    if settings_row:
        try:
            settings = json.loads(settings_row['value'])
        except (TypeError, ValueError):
            settings = None
    cells = {}
    for row in cell_rows:
        cells.setdefault(row['schedule_id'], []).append((row['person'], row['day_col'], bool(row['assigned'])))
    for schedule in schedules:
        schedule['cells'] = cells.get(schedule.pop('id'), [])
    return (settings if isinstance(settings, dict) else None), schedules


def restore_owner_data(backup, owner=""):
    """Doğrulanmış yedeği (nobet_core.parse_backup) kullanıcının verisine yazar.

    Yedekteki aylar mevcut kayıtların üzerine yazılır, yedekte olmayan aylara
    dokunulmaz. Hepsi tek işlemde yazılır: bir hata olursa hiçbir şey
    değişmez. Hata olursa yukarı iletilir.
    """
    conn = None
    try:
        conn = _connect()
        with conn:
            with conn.cursor() as cur:
                for item in backup.get("schedules") or []:
                    cur.execute('''
                        INSERT INTO schedules (name, year, month, team_members, pref_json, settings_json, rows_liste_json, owner, updated_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                        ON CONFLICT (owner, name, year, month) DO UPDATE
                            SET updated_at = CURRENT_TIMESTAMP,
                                team_members = EXCLUDED.team_members,
                                pref_json = EXCLUDED.pref_json,
                                settings_json = EXCLUDED.settings_json,
                                rows_liste_json = EXCLUDED.rows_liste_json
                        RETURNING id
                    ''', (
                        item["name"], item["year"], item["month"],
                        json.dumps(item["team"], ensure_ascii=False),
                        json.dumps(item["preferences"], ensure_ascii=False) if item["preferences"] is not None else None,
                        json.dumps(item["settings"], ensure_ascii=False) if item["settings"] is not None else None,
                        json.dumps(item["rows"], ensure_ascii=False) if item["rows"] is not None else None,
                        owner,
                    ))
                    schedule_id = cur.fetchone()[0]
                    cur.execute('DELETE FROM schedule_data WHERE schedule_id = %s', (schedule_id,))
                    assigned = set(item["assigned"])
                    rows = [
                        (schedule_id, person, day, (person, day) in assigned)
                        for person in item["team"] for day in item["days"]
                    ]
                    if rows:
                        execute_values(cur, '''
                            INSERT INTO schedule_data (schedule_id, person, day_col, assigned)
                            VALUES %s
                        ''', rows)
                if backup.get("settings") is not None:
                    cur.execute('''
                        INSERT INTO app_settings (key, value, updated_at)
                        VALUES (%s, %s, CURRENT_TIMESTAMP)
                        ON CONFLICT (key) DO UPDATE
                            SET value = EXCLUDED.value,
                                updated_at = CURRENT_TIMESTAMP
                    ''', (settings_key(owner), json.dumps(backup["settings"], ensure_ascii=False)))
        return True
    finally:
        if conn is not None:
            conn.close()
