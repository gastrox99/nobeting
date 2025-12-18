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

## Recent Changes
- **Fixed**: AYB/GYB assignments now stable when switching person in personalized view
  - Issue: Random shuffle was re-running on every selectbox change
  - Solution: Added session state caching for rows_liste and ayb_counts
  - Only regenerates when schedule actually changes (detected via hash)

## Project Structure
- `nobet.py` - Main application file
- `requirements.txt` - Python dependencies
