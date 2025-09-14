@echo off
echo Starting PDF Q&A Chatbot...
echo.
echo Choose mode:
echo 1. Development (with auto-reload)
echo 2. Production (no auto-reload)
echo.
set /p choice="Enter choice (1 or 2): "

if "%choice%"=="1" (
    echo Starting in Development Mode...
    python run_webapp.py
) else if "%choice%"=="2" (
    echo Starting in Production Mode...
    python run_production.py
) else (
    echo Invalid choice. Starting in Development Mode...
    python run_webapp.py
)

pause
