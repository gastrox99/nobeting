# Adil Nöbet (Fair Shift Scheduling)

## Overview
A Streamlit-based shift/duty scheduling application (Turkish: "Nöbet Yönetimi") that helps teams fairly distribute shifts among team members using AI simulation.

## Features
- Team member management
- Availability input (days off, holidays)
- AI-powered fair shift distribution (100 simulations)
- Visual schedule editor
- Work load analysis and statistics
- Pay/compensation calculations
- Export to CSV and PNG

## Tech Stack
- **Language**: Python 3.11
- **Framework**: Streamlit
- **Dependencies**: pandas, numpy, matplotlib, tabulate

## Running the App
The app runs on port 5000 via the Streamlit workflow:
```bash
streamlit run nobet.py --server.port 5000 --server.address 0.0.0.0 --server.headless true --browser.gatherUsageStats false
```

## Recent Changes (Dec 2024)

### Bug Fixes Round 2
- **Fixed**: Forbidden pairs parsing now supports both comma and newline separators
- **Fixed**: Session state sync - schedule now auto-adjusts when team or month changes
- **Fixed**: Daily list now shows ALL assigned people (not just first 2) when kişi_sayısı > 2
- **Fixed**: Personal detail view now correctly shows shifts for all roles (not just AYB/GYB)
- **Fixed**: Validation function now properly receives kişi_sayısı parameter
- **Fixed**: Pair matrix counts ALL pairs correctly (not just first 2 people)

### Bug Fixes Round 1
- **Fixed**: Data editor stability with form-based editing (prevents continuous reruns)
- **Fixed**: Arrow serialization errors in pair_matrix with consistent data types

## Latest Improvements - Step 1 Complete ✅

### 1. Database & Save/Load Functionality (IMPLEMENTED)
- **New**: PostgreSQL database integration for schedule persistence
- **Save**: Click "Kaydet/Yükle" → "Kaydet" tab, name your schedule, click save
- **Load**: "Kaydet/Yükle" → "Yükle" tab, select schedule, click load
- **List**: View all saved schedules with timestamps, delete old ones
- **Benefit**: Schedules no longer lost on refresh!

### 2. Input Validation (IMPLEMENTED) ✅
- **Prevents**: Empty teams, negative pay, duplicate names, invalid holidays
- **Feasibility checks**: Warns if too many shifts per person or too few positions
- **Error messages**: Clear Turkish messages explaining what's wrong
- **Warnings**: Shows alerts for edge cases (0 pay, overloaded team, underutilized team)
- **Benefit**: No more crashes on bad data - user gets helpful feedback

### 3. Better Shift Editor UI (IMPLEMENTED) ✅
- **Day Range Selector**: Choose start/end days with sliders instead of horizontal scrolling
- **Mini-stats**: Shows total assignments, day range, column count for selected days
- **Focused Editing**: Edit 10 days at a time instead of scrolling through 30 columns
- **Benefit**: Much easier to work with large schedules!

### Remaining (Not Implemented):
- Priority 4: Advanced features (swap shifts, templates, preferences)

## Project Structure
- `nobet.py` - Main application file
- `db.py` - Database functions (PostgreSQL backend)
- `requirements.txt` - Python dependencies
