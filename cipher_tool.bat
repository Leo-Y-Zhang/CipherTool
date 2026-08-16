@echo off
REM ===================================================================
REM  cipher_tool -- one-click launcher for Windows.
REM
REM  Double-click this file to open the interactive shell.
REM  Or run it from a terminal with arguments:
REM      cipher_tool.bat analyse message.txt
REM      cipher_tool.bat auto message.txt --fast
REM
REM  Nothing needs installing. There are no dependencies.
REM ===================================================================
setlocal
cd /d "%~dp0"

set "PY="
where py >nul 2>&1 && set "PY=py -3"
if not defined PY where python >nul 2>&1 && set "PY=python"
if not defined PY where python3 >nul 2>&1 && set "PY=python3"
if not defined PY goto no_python

%PY% -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
if errorlevel 1 goto old_python

REM Run straight from src/ so the toolkit works without being installed.
set "PYTHONPATH=%~dp0src"

if "%~1"=="" goto interactive

%PY% -m cipher_tool %*
exit /b %errorlevel%

:interactive
REM Default to the paste-and-solve flow. It is what someone wants in the
REM first thirty seconds, and unlike the command shell it copes with a
REM ciphertext pasted across several lines.
%PY% -m cipher_tool paste
echo.
pause
exit /b 0

:no_python
echo.
echo   Python was not found on this computer.
echo.
echo   cipher_tool needs Python 3.10 or newer. Install it from:
echo     https://www.python.org/downloads/
echo.
echo   On the first installer screen, TICK "Add python.exe to PATH",
echo   then run this file again.
echo.
pause
exit /b 1

:old_python
echo.
echo   The Python on this computer is too old.
echo.
%PY% -c "import sys; print('   Found version ' + sys.version.split()[0] + ', but 3.10 or newer is needed.')"
echo.
echo   Install a newer Python from https://www.python.org/downloads/
echo.
pause
exit /b 1
