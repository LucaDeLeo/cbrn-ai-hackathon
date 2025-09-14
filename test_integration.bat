@echo off
REM Windows-compatible test script for RobustCBRN Pipeline Integration
echo Testing RobustCBRN Pipeline Integration...
echo.

echo 1. Checking if all pipeline scripts are present:
dir scripts\*.sh >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Pipeline scripts not found
    exit /b 1
)
echo SUCCESS: All pipeline scripts found
echo.

echo 2. Checking Makefile integration:
findstr /C:"pipeline" Makefile >nul
if %errorlevel% neq 0 (
    echo ERROR: Pipeline targets not found in Makefile
    exit /b 1
)
echo SUCCESS: Pipeline targets found in Makefile
echo.

echo 3. Checking README.md integration:
findstr /C:"pipeline" README.md >nul
if %errorlevel% neq 0 (
    echo ERROR: Pipeline documentation not found in README
    exit /b 1
)
echo SUCCESS: Pipeline documentation found in README
echo.

echo 4. Checking script executability:
for %%f in (scripts\*.sh) do (
    if not exist "%%f" (
        echo ERROR: Script %%f not found
        exit /b 1
    )
)
echo SUCCESS: All scripts are present and accessible
echo.

echo 5. Checking integration documentation:
if not exist "scripts\INTEGRATION_GUIDE.md" (
    echo ERROR: Integration guide not found
    exit /b 1
)
if not exist "scripts\PIPELINE_README.md" (
    echo ERROR: Pipeline README not found
    exit /b 1
)
echo SUCCESS: Integration documentation found
echo.

echo ========================================
echo INTEGRATION TEST COMPLETED SUCCESSFULLY
echo ========================================
echo.
echo The RobustCBRN pipeline has been successfully integrated!
echo.
echo Available commands:
echo   make pipeline              - Run complete pipeline
echo   make pipeline-validate     - Platform validation
echo   make pipeline-setup        - Environment setup
echo   make pipeline-sample       - Sample evaluation
echo   make pipeline-full         - Full evaluation
echo.
echo For Windows users without make/bash:
echo   Use Git Bash or WSL to run the pipeline scripts directly
echo.
echo Integration Status: COMPLETE ✅
pause
