---
name: Python package install side effects
description: Replit Python dependency installation may modify requirements and restart workflows unexpectedly
---

After installing a Python package through the workspace package manager, inspect the dependency file and the web workflow before finishing.

**Why:** In September 2026, installing a package already listed in requirements appended it again along with an unrelated test package. The automatic workflow restart also failed because the prior Streamlit process was still occupying port 5000; the code and tests were healthy.

**How to apply:** Remove duplicate or unrelated dependency entries, then check whether the managed web workflow is running. If a port conflict occurs, identify the existing listener before restarting rather than changing the app's configured port.