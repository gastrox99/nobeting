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
- **Fixed**: Data editor stability and manual shift editing
  - Issue 1: AYB/GYB assignments were regenerating on every selectbox change
    - Solution: Added session state caching for rows_liste and ayb_counts
  - Issue 2: Manual edits were regenerating assignments and throwing back to first column
    - Solution: Changed regeneration logic to only trigger on AI button click
    - Added `should_regenerate_assignments` flag to control regeneration
    - Data editor now maintains state with key="schedule_editor"
    - Used .copy() to prevent reference mutation issues
  - Result: Smooth manual editing experience, stable view, no double-clicks needed

- **Fixed**: Arrow serialization error in pair_matrix
  - Issue: Mixed int/string types in pair_matrix caused Arrow serialization failures
  - Solution: Changed initialization from `pd.DataFrame(0, ...)` to `pd.DataFrame('', ..., dtype=object)`
  - Updated increment logic to handle type conversion: `int(pair_matrix.loc[x,y] or 0) + 1`
  - Result: No more Arrow type errors when displaying the matching matrix

## Project Structure
- `nobet.py` - Main application file
- `requirements.txt` - Python dependencies
