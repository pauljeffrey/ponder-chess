#!/bin/sh
python scripts/fetch_weights.py
exec uvicorn server.app:app --host 0.0.0.0 --port 8000
