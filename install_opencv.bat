@echo off
chcp 65001 >nul
setlocal

echo ============================================================
echo   OpenCV One-Click Installer
echo   Target: opencv-python 4.14.0.94 + numpy
echo ============================================================
echo.

rem ---- Step 0: check Python ----
where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python was not found on PATH.
    echo Please install Python 3.11 - 3.13 from https://www.python.org/downloads/
    echo IMPORTANT: check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

echo [0/4] Python found:
python --version
echo.

rem ---- Step 1: upgrade pip ----
echo [1/4] Upgrading pip ...
python -m pip install --upgrade pip --disable-pip-version-check
echo.

rem ---- Step 2: install OpenCV and NumPy ----
echo [2/4] Installing opencv-python 4.14.0.94 and numpy ...
echo        (first run downloads about 60 MB, please wait)
python -m pip install "opencv-python==4.14.0.94" numpy --disable-pip-version-check
if errorlevel 1 (
    echo.
    echo [ERROR] Installation failed.
    echo Try the Tsinghua mirror instead:
    echo   python -m pip install "opencv-python==4.14.0.94" numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
    echo.
    pause
    exit /b 1
)
echo.

rem ---- Step 3: print version ----
echo [3/4] Installed version:
python -c "import cv2, numpy; print('  OpenCV:', cv2.__version__); print('  NumPy :', numpy.__version__)"
echo.

rem ---- Step 4: run the full check ----
echo [4/4] Running the verification script ...
echo.
python "%~dp0check_cv.py"

echo.
echo ============================================================
echo   Done. If you see "all checks passed", you are ready.
echo ============================================================
pause
