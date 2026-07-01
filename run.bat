@echo off
chcp 65001 >nul

echo ========================================
echo    Image Matting Tool - Starting...
echo ========================================
echo.

REM Check for virtual environment, prefer it
if exist "venv\Scripts\python.exe" (
    set PYTHON_EXE=venv\Scripts\python.exe
    echo [info] using venv python
) else if exist ".venv\Scripts\python.exe" (
    set PYTHON_EXE=.venv\Scripts\python.exe
    echo [info] using .venv python
) else (
    set PYTHON_EXE=python
    echo [info] using system python
)

echo [step 1/2] checking and installing dependencies...
"%PYTHON_EXE%" -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
if %errorlevel% neq 0 (
    echo.
    echo [error] dependency install failed, retrying with default mirror...
    "%PYTHON_EXE%" -m pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo.
        echo [error] dependency install failed. check your network connection.
        pause
        exit /b 1
    )
)

echo.
echo [step 2/2] launching application...
echo.

"%PYTHON_EXE%" main.py
if %errorlevel% neq 0 (
    echo.
    echo [error] application exited abnormally, code: %errorlevel%
    pause
    exit /b %errorlevel%
)

pause
