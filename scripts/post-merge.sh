#!/usr/bin/env bash
set -euo pipefail

# Replit, requirements.txt bağımlılıklarını proje ortamındaki .pythonlibs
# dizininde yönetir; burada immutable sistem Python'una kurulum yapılmaz.
python -m py_compile nobet.py nobet_core.py
python -m pytest test_nobet.py -q --tb=short