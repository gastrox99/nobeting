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

### New Features
- **AI Simulation**: 100 random simulations to find the most balanced schedule
- **streamlit-aggrid Preference Grid**: Interactive colored grid for preference input (click cells to edit, color-coded display)
- **Custom Role Names**: Define your own shift role names (e.g., "Nöbetçi, Yardımcı" instead of default AYB/GYB)
- **Min/Max Limits**: Set individual shift limits per person (e.g., "Ali:5-10")
- **Weekend Balance**: Algorithm avoids assigning consecutive weekend shifts
- **Excel Export**: Download schedule as formatted Excel file with multiple sheets
- **Auto-Save**: Automatically saves to database every 30 seconds
- **Visual Calendar Preferences**: Click-to-select color-coded calendar for each person (green=preferred, yellow=avoid, red=unavailable)
- **Print-Friendly View**: Download HTML file optimized for printing
- **Undo/Redo**: Revert manual edits with history (up to 10 states)

### Bug Fixes Round 3 (Dec 25, 2024)
- **Fixed**: Removed unused st_aggrid import (faster loading)
- **Fixed**: kişi_sayısı now passed as parameter to scheduling function (no global scope dependency)
- **Fixed**: Team mismatch warning when loading saved schedules with different people
- **Fixed**: Mobile scroll hint now hidden on desktop (CSS media query)
- **Changed**: Default role names from "Kişi1, Kişi2" to "Görev1, Görev2"
- **Removed**: "HS" (hafta sonu) column from workload analysis table

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

### 4. Mobile Responsive Design (IMPLEMENTED) ✅
- **Settings Panel**: Columns stack vertically on mobile (<768px)
- **Schedule Grid**: Horizontal scroll on mobile with touch support
- **Button Sizes**: Optimized for touch on small screens
- **Text Sizes**: Reduced for better mobile readability
- **Tablet Support**: 2-column layout for medium screens

### Remaining (Not Implemented):
- Priority 5: Advanced features (swap shifts, templates, preferences)

## Project Structure
- `nobet.py` - Main application file
- `db.py` - Database functions (PostgreSQL backend)
- `requirements.txt` - Python dependencies
