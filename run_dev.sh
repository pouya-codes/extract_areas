#!/bin/bash
# Development server with hot-reload
# This script starts the FastAPI server with automatic reload on code changes

echo "🚀 Starting FastAPI development server with hot-reload..."
echo "📁 Watching: current directory, models/, src/"
echo "🌐 Server: http://localhost:8000"
echo "📖 API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Run uvicorn with reload
python -m uvicorn app_refactored:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload \
    --reload-dir . \
    --reload-dir models \
    --reload-dir src \
    --log-level info
