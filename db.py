# -*- coding: utf-8 -*-
import os
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
import pandas as pd
from datetime import datetime
import json

DATABASE_URL = os.getenv('DATABASE_URL')

def init_db():
    """Initialize database schema"""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        
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
        
        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception as e:
        print(f"Database init error: {e}")
        return False

def save_schedule(name, year, month, team_members, schedule_df, pref_df=None, settings_dict=None):
    """Save a schedule to database"""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        
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
        
        # Insert or update schedule
        cur.execute('''
            INSERT INTO schedules (name, year, month, team_members, pref_json, settings_json, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
            ON CONFLICT (name, year, month) DO UPDATE
                SET updated_at = CURRENT_TIMESTAMP,
                    team_members = EXCLUDED.team_members,
                    pref_json = EXCLUDED.pref_json,
                    settings_json = EXCLUDED.settings_json
            RETURNING id
        ''', (name, year, month, team_str, pref_json, settings_json_str))
        
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
        
        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception as e:
        print(f"Save schedule error: {e}")
        return False

def load_schedule(name, year, month):
    """Load a schedule from database.
    Returns (team_members, schedule_df, pref_df, settings_dict).
    pref_df and settings_dict are None when not saved.
    """
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        # Get schedule
        cur.execute('''
            SELECT * FROM schedules WHERE name = %s AND year = %s AND month = %s
        ''', (name, year, month))
        
        schedule_row = cur.fetchone()
        if not schedule_row:
            return None, None, None, None
        
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
                pref_df = pd.read_json(raw_pref)
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
        
        # Get schedule data
        cur.execute('''
            SELECT person, day_col, assigned FROM schedule_data 
            WHERE schedule_id = %s ORDER BY person, day_col
        ''', (schedule_id,))
        
        data_rows = cur.fetchall()
        cur.close()
        conn.close()
        
        if not data_rows:
            return team_members, None, pref_df, settings_dict
        
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
        return team_members, df, pref_df, settings_dict
        
    except Exception as e:
        print(f"Load schedule error: {e}")
        return None, None, None, None

def list_schedules():
    """List all saved schedules"""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        cur.execute('''
            SELECT name, year, month, updated_at FROM schedules 
            ORDER BY updated_at DESC
        ''')
        
        schedules = cur.fetchall()
        cur.close()
        conn.close()
        return schedules
    except Exception as e:
        print(f"List schedules error: {e}")
        return []

def delete_schedule(name, year, month):
    """Delete a schedule"""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        
        cur.execute('''
            DELETE FROM schedules WHERE name = %s AND year = %s AND month = %s
        ''', (name, year, month))
        
        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception as e:
        print(f"Delete schedule error: {e}")
        return False
