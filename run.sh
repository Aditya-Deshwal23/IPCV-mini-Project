#!/usr/bin/env sh
set -eu
exec python -m streamlit run app.py "$@"
