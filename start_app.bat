@echo off
title AI Agent Builder Launcher
echo Starting AI System servers...

echo [1/2] Launching Python Backend in a new window...
start "Backend Server (FastAPI)" cmd /k "python -m uvicorn backend.main:app --reload --port 8000"

echo [2/2] Launching React Frontend in a new window...
start "Frontend Server (Vite)" cmd /k "cd frontend && npm run dev"

echo.
echo =======================================================
echo ✅ SERVERS LAUNCHED!
echo -------------------------------------------------------
echo Two new terminal windows should have popped up.
echo Application UI: http://localhost:5173
echo =======================================================
echo.
echo You can now close this launcher window.
pause
