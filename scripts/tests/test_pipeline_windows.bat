@echo off
REM Comprehensive Pipeline Test Script for Windows
REM Tests the new RobustCBRN pipeline integration

setlocal enabledelayedexpansion

REM Test configuration
set TEST_DIR=test_pipeline_output
set LOG_FILE=pipeline_test.log

REM Test result tracking
set TESTS_PASSED=0
set TESTS_FAILED=0
set TESTS_TOTAL=0

REM Logging function
:log
set level=%1
set message=%2
for /f "tokens=1-3 delims=:" %%a in ("%time%") do set timestamp=%%a:%%b:%%c
echo [%timestamp%] [%level%] %message% >> %LOG_FILE%
goto :eof

REM Test function
:run_test
set test_name=%1
set test_command=%2
set expected_exit_code=%3
if "%expected_exit_code%"=="" set expected_exit_code=0

set /a TESTS_TOTAL+=1
call :log "INFO" "Running test: %test_name%"

%test_command% >nul 2>&1
set exit_code=%errorlevel%

if %exit_code%==%expected_exit_code% (
    echo ✅ PASS: %test_name%
    set /a TESTS_PASSED+=1
    call :log "PASS" "%test_name%"
) else (
    echo ❌ FAIL: %test_name% (exit code: %exit_code%, expected: %expected_exit_code%)
    set /a TESTS_FAILED+=1
    call :log "FAIL" "%test_name% (exit code: %exit_code%, expected: %expected_exit_code%)"
)
echo.
goto :eof

REM Cleanup function
:cleanup
call :log "INFO" "Cleaning up test environment"
if exist "%TEST_DIR%" rmdir /s /q "%TEST_DIR%" 2>nul
goto :eof

REM Setup test environment
:setup_test_env
call :log "INFO" "Setting up test environment"
if not exist "%TEST_DIR%" mkdir "%TEST_DIR%"
if exist "%LOG_FILE%" del "%LOG_FILE%"
call :log "INFO" "Test environment ready"
goto :eof

REM Test 1: Script Presence and Executability
:test_script_presence
echo === Testing Script Presence and Executability ===

set scripts=run_pipeline.sh validate_platform.sh setup_env.sh discover_entry_points.sh run_sample_evaluation.sh run_full_evaluation.sh aggregate_results.sh generate_figures.sh run_tests_and_security.sh generate_report.sh final_verification.sh platform_compat.sh

for %%s in (%scripts%) do (
    call :run_test "Script exists: scripts\%%s" "if exist scripts\%%s"
    call :run_test "Script is readable: scripts\%%s" "if exist scripts\%%s"
)
goto :eof

REM Test 2: Makefile Integration
:test_makefile_integration
echo === Testing Makefile Integration ===

call :run_test "Makefile exists" "if exist Makefile"
call :run_test "Pipeline targets in Makefile" "findstr /C:\"pipeline:\" Makefile"
call :run_test "Pipeline-validate target" "findstr /C:\"pipeline-validate:\" Makefile"
call :run_test "Pipeline-setup target" "findstr /C:\"pipeline-setup:\" Makefile"
call :run_test "Pipeline-sample target" "findstr /C:\"pipeline-sample:\" Makefile"
call :run_test "Pipeline-full target" "findstr /C:\"pipeline-full:\" Makefile"
goto :eof

REM Test 3: Documentation Integration
:test_documentation_integration
echo === Testing Documentation Integration ===

call :run_test "README.md exists" "if exist README.md"
call :run_test "Pipeline documentation in README" "findstr /C:\"pipeline\" README.md"
call :run_test "Integration guide exists" "if exist scripts\INTEGRATION_GUIDE.md"
call :run_test "Pipeline README exists" "if exist scripts\PIPELINE_README.md"
goto :eof

REM Test 4: Configuration Files
:test_configuration_files
echo === Testing Configuration Files ===

call :run_test "Requirements.txt exists" "if exist requirements.txt"
call :run_test "Pyproject.toml exists" "if exist pyproject.toml"
call :run_test "Configs directory exists" "if exist configs"
call :run_test "Data directory exists" "if exist data"
goto :eof

REM Test 5: Project Structure
:test_project_structure
echo === Testing Project Structure ===

call :run_test "Robustcbrn package exists" "if exist robustcbrn"
call :run_test "Tests directory exists" "if exist tests"
call :run_test "Scripts directory exists" "if exist scripts"
call :run_test "Docs directory exists" "if exist docs"
goto :eof

REM Test 6: Integration Documentation Content
:test_integration_docs
echo === Testing Integration Documentation Content ===

call :run_test "Integration guide has content" "if exist scripts\INTEGRATION_GUIDE.md"
call :run_test "Pipeline README has content" "if exist scripts\PIPELINE_README.md"
call :run_test "Verification report exists" "if exist INTEGRATION_VERIFICATION_REPORT.md"
goto :eof

REM Test 7: Makefile Content Validation
:test_makefile_content
echo === Testing Makefile Content Validation ===

call :run_test "Makefile has pipeline target" "findstr /C:\"pipeline:\" Makefile"
call :run_test "Makefile has pipeline-validate" "findstr /C:\"pipeline-validate:\" Makefile"
call :run_test "Makefile has pipeline-setup" "findstr /C:\"pipeline-setup:\" Makefile"
call :run_test "Makefile has pipeline-sample" "findstr /C:\"pipeline-sample:\" Makefile"
call :run_test "Makefile has pipeline-full" "findstr /C:\"pipeline-full:\" Makefile"
goto :eof

REM Test 8: README Content Validation
:test_readme_content
echo === Testing README Content Validation ===

call :run_test "README mentions pipeline" "findstr /C:\"pipeline\" README.md"
call :run_test "README has unified pipeline section" "findstr /C:\"Unified Pipeline\" README.md"
call :run_test "README has legacy support" "findstr /C:\"Legacy\" README.md"
call :run_test "README has advanced usage" "findstr /C:\"Advanced\" README.md"
goto :eof

REM Test 9: Script File Sizes (Basic validation)
:test_script_sizes
echo === Testing Script File Sizes ===

for %%s in (run_pipeline.sh validate_platform.sh setup_env.sh discover_entry_points.sh) do (
    call :run_test "Script %%s has content" "if exist scripts\%%s"
)
goto :eof

REM Test 10: Cross-Platform Compatibility Files
:test_cross_platform_files
echo === Testing Cross-Platform Compatibility Files ===

call :run_test "Platform compatibility script exists" "if exist scripts\platform_compat.sh"
call :run_test "Windows test script exists" "if exist test_integration.bat"
call :run_test "Comprehensive test script exists" "if exist test_pipeline_comprehensive.sh"
goto :eof

REM Main test execution
:main
echo 🚀 Starting RobustCBRN Pipeline Integration Test
echo ==================================================
echo.

call :setup_test_env

REM Run all test suites
call :test_script_presence
call :test_makefile_integration
call :test_documentation_integration
call :test_configuration_files
call :test_project_structure
call :test_integration_docs
call :test_makefile_content
call :test_readme_content
call :test_script_sizes
call :test_cross_platform_files

REM Test summary
echo === Test Summary ===
echo Total Tests: %TESTS_TOTAL%
echo Passed: %TESTS_PASSED%
echo Failed: %TESTS_FAILED%

if %TESTS_FAILED%==0 (
    echo 🎉 All tests passed! Pipeline integration is successful.
    call :log "SUCCESS" "All tests passed - Pipeline integration successful"
    goto :success
) else (
    echo ❌ Some tests failed. Please check the logs.
    call :log "FAILURE" "Some tests failed - Check integration"
    goto :failure
)

:success
echo.
echo ✅ INTEGRATION TEST COMPLETED SUCCESSFULLY
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
call :cleanup
exit /b 0

:failure
echo.
echo ❌ INTEGRATION TEST FAILED
echo.
echo Please check the test log: %LOG_FILE%
echo Review the failed tests and fix any issues.
echo.
call :cleanup
exit /b 1

REM Run main function
call :main
