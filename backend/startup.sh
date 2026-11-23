#!/bin/bash

# Run database migrations and seeding first
python -m app.core.database_seed

# Then start the application
# --timeout 600: LLM transcription requests (multi-page PDFs, long text) can take minutes.
gunicorn -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 --workers 4 --worker-class uvicorn.workers.UvicornWorker --timeout 600 app.main:app
