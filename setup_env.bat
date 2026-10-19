@echo off
REM ============================================================
REM  One-click setup: create venv + install OpenCV + verify
REM  Usage: double-click this file, or drag it into a cmd window
REM  NOTE: all messages in English on purpose -- cmd decodes .bat
REM        as GBK, Chinese text would show as garbage.
REM ============================================================

set "PROJDIR=%~dp0"
set "VENV=%PROJDIR%cv_env"

echo.
echo ============================================================
echo   STEP 1 / 4   Locating Python
echo ============================================================

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found in PATH.
    echo         If you use Anaconda, run this from "Anaconda Prompt".
    echo         Or install Python from https://www.python.org/downloads/
    pause
    exit /b 1
)

python --version
echo.

echo ============================================================
echo   STEP 2 / 4   Creating virtual environment: cv_env
echo ============================================================

if exist "%VENV%\Scripts\python.exe" (
    echo Found existing cv_env -- skipping creation, will just install into it.
) else (
    python -m venv "%VENV%"
    if errorlevel 1 (
        echo [ERROR] Failed to create venv. See message above.
        pause
        exit /b 1
    )
    echo Created.
)
echo.

echo ============================================================
echo   STEP 3 / 4   Installing packages into cv_env
echo   (opencv 4.14.0.94 + numpy + matplotlib)
echo ============================================================

call "%VENV%\Scripts\python.exe" -m pip install --upgrade pip --disable-pip-version-check
if errorlevel 1 (
    echo [WARN] pip upgrade failed -- continuing anyway.
)

echo.
echo Trying default PyPI source first...
call "%VENV%\Scripts\python.exe" -m pip install opencv-python==4.14.0.94 numpy matplotlib --disable-pip-version-check
if errorlevel 1 (
    echo.
    echo Default source failed. Retrying with Tsinghua mirror...
    call "%VENV%\Scripts\python.exe" -m pip install opencv-python==4.14.0.94 numpy matplotlib ^
        -i https://pypi.tuna.tsinghua.edu.cn/simple --disable-pip-version-check
)

if errorlevel 1 (
    echo.
    echo Retrying with Aliyun mirror...
    call "%VENV%\Scripts\python.exe" -m pip install opencv-python==4.14.0.94 numpy matplotlib ^
        -i https://mirrors.aliyun.com/pypi/simple/ --disable-pip-version-check
)
echo.

echo ============================================================
echo   STEP 4 / 4   Verification
echo ============================================================

call "%VENV%\Scripts\python.exe" "%PROJDIR%check_cv.py"
set RC=%errorlevel%

echo.
echo ============================================================
if "%RC%"=="0" (
    echo   DONE - environment is ready.
    echo.
    echo   Your venv Python is at:
    echo     %VENV%\Scripts\python.exe
    echo.
    echo   From now on, activate it FIRST in every new terminal:
    echo     %VENV%\Scripts\activate
    echo   You will see  (cv_env)  at the front of the prompt.
    echo.
    echo   In VS Code: Ctrl+Shift+P ^-^> "Python: Select Interpreter"
    echo              ^-^> pick  %VENV%\Scripts\python.exe
) else (
    echo   Something went wrong. Copy the red text above and send it.
)
echo ============================================================
echo.
pause
