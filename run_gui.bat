@echo off
REM Video Highlights Generator - GUI Launcher for Windows
REM This script launches the Tkinter GUI on Windows

echo ============================================
echo Video Highlights Generator - GUI Launcher
echo ============================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from python.org
    pause
    exit /b 1
)

echo Starting GUI...
echo.

REM Launch the GUI
python VideoHighlightsGUI.py

if errorlevel 1 (
    echo.
    echo ERROR: Failed to start GUI
    echo Check that all dependencies are installed: pip install -r requirements.txt
    pause
)
