@echo off
REM Windows batch script to run the sensor localization example

echo ==========================================
echo Decentralized Sensor Network Localization
echo ==========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python from python.org
    pause
    exit /b 1
)

REM Check/Install dependencies
echo Checking dependencies...
pip show numpy >nul 2>&1
if errorlevel 1 (
    echo Installing required packages...
    pip install -r requirements.txt
)

REM Run the example
echo.
echo Running sensor localization example...
echo.
python simple_example.py

echo.
echo ==========================================
echo Example complete! Check the generated images:
echo - simple_example_results.png
echo - simple_example_convergence.png
echo ==========================================
echo.
pause