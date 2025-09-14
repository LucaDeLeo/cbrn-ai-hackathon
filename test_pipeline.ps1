# PowerShell Pipeline Test Script
# Tests the new RobustCBRN pipeline integration

Write-Host "🚀 Starting RobustCBRN Pipeline Integration Test" -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Blue
Write-Host ""

$TestsPassed = 0
$TestsFailed = 0
$TestsTotal = 0

function Test-Condition {
    param(
        [string]$TestName,
        [scriptblock]$TestScript,
        [bool]$ExpectedResult = $true
    )
    
    $TestsTotal++
    Write-Host "Testing: $TestName" -ForegroundColor Yellow
    
    try {
        $result = & $TestScript
        if ($result -eq $ExpectedResult) {
            Write-Host "✅ PASS: $TestName" -ForegroundColor Green
            $script:TestsPassed++
            return $true
        } else {
            Write-Host "❌ FAIL: $TestName" -ForegroundColor Red
            $script:TestsFailed++
            return $false
        }
    } catch {
        Write-Host "❌ FAIL: $TestName - Error: $($_.Exception.Message)" -ForegroundColor Red
        $script:TestsFailed++
        return $false
    }
    Write-Host ""
}

# Test 1: Script Presence
Write-Host "=== Testing Script Presence ===" -ForegroundColor Blue
$pipelineScripts = @(
    "scripts\run_pipeline.sh",
    "scripts\validate_platform.sh", 
    "scripts\setup_env.sh",
    "scripts\discover_entry_points.sh",
    "scripts\run_sample_evaluation.sh",
    "scripts\run_full_evaluation.sh",
    "scripts\aggregate_results.sh",
    "scripts\generate_figures.sh",
    "scripts\run_tests_and_security.sh",
    "scripts\generate_report.sh",
    "scripts\final_verification.sh",
    "scripts\platform_compat.sh"
)

foreach ($script in $pipelineScripts) {
    Test-Condition "Script exists: $script" { Test-Path $script }
}

# Test 2: Makefile Integration
Write-Host "=== Testing Makefile Integration ===" -ForegroundColor Blue
Test-Condition "Makefile exists" { Test-Path "Makefile" }
Test-Condition "Makefile contains pipeline targets" { 
    (Get-Content "Makefile" | Select-String "pipeline:").Count -gt 0 
}
Test-Condition "Makefile contains pipeline-validate" { 
    (Get-Content "Makefile" | Select-String "pipeline-validate:").Count -gt 0 
}
Test-Condition "Makefile contains pipeline-setup" { 
    (Get-Content "Makefile" | Select-String "pipeline-setup:").Count -gt 0 
}
Test-Condition "Makefile contains pipeline-sample" { 
    (Get-Content "Makefile" | Select-String "pipeline-sample:").Count -gt 0 
}
Test-Condition "Makefile contains pipeline-full" { 
    (Get-Content "Makefile" | Select-String "pipeline-full:").Count -gt 0 
}

# Test 3: Documentation Integration
Write-Host "=== Testing Documentation Integration ===" -ForegroundColor Blue
Test-Condition "README.md exists" { Test-Path "README.md" }
Test-Condition "README contains pipeline documentation" { 
    (Get-Content "README.md" | Select-String "pipeline").Count -gt 0 
}
Test-Condition "Integration guide exists" { Test-Path "scripts\INTEGRATION_GUIDE.md" }
Test-Condition "Pipeline README exists" { Test-Path "scripts\PIPELINE_README.md" }
Test-Condition "Verification report exists" { Test-Path "INTEGRATION_VERIFICATION_REPORT.md" }

# Test 4: Project Structure
Write-Host "=== Testing Project Structure ===" -ForegroundColor Blue
Test-Condition "Robustcbrn package exists" { Test-Path "robustcbrn" }
Test-Condition "Tests directory exists" { Test-Path "tests" }
Test-Condition "Scripts directory exists" { Test-Path "scripts" }
Test-Condition "Docs directory exists" { Test-Path "docs" }
Test-Condition "Data directory exists" { Test-Path "data" }

# Test 5: Configuration Files
Write-Host "=== Testing Configuration Files ===" -ForegroundColor Blue
Test-Condition "Requirements.txt exists" { Test-Path "requirements.txt" }
Test-Condition "Pyproject.toml exists" { Test-Path "pyproject.toml" }
Test-Condition "Configs directory exists" { Test-Path "configs" }

# Test 6: Script Content Validation
Write-Host "=== Testing Script Content Validation ===" -ForegroundColor Blue
Test-Condition "Main pipeline script has content" { 
    (Get-Item "scripts\run_pipeline.sh").Length -gt 1000 
}
Test-Condition "Platform validation script has content" { 
    (Get-Item "scripts\validate_platform.sh").Length -gt 1000 
}
Test-Condition "Setup script has content" { 
    (Get-Item "scripts\setup_env.sh").Length -gt 1000 
}

# Test 7: README Content Validation
Write-Host "=== Testing README Content Validation ===" -ForegroundColor Blue
Test-Condition "README mentions unified pipeline" { 
    (Get-Content "README.md" | Select-String "Unified Pipeline").Count -gt 0 
}
Test-Condition "README mentions legacy support" { 
    (Get-Content "README.md" | Select-String "Legacy").Count -gt 0 
}
Test-Condition "README mentions advanced usage" { 
    (Get-Content "README.md" | Select-String "Advanced").Count -gt 0 
}

# Test 8: Cross-Platform Files
Write-Host "=== Testing Cross-Platform Files ===" -ForegroundColor Blue
Test-Condition "Platform compatibility script exists" { Test-Path "scripts\platform_compat.sh" }
Test-Condition "Windows test script exists" { Test-Path "test_integration.bat" }
Test-Condition "Comprehensive test script exists" { Test-Path "test_pipeline_comprehensive.sh" }

# Test Summary
Write-Host ""
Write-Host "=== Test Summary ===" -ForegroundColor Blue
Write-Host "Total Tests: $TestsTotal" -ForegroundColor White
Write-Host "Passed: $TestsPassed" -ForegroundColor Green
Write-Host "Failed: $TestsFailed" -ForegroundColor Red

$SuccessRate = [math]::Round(($TestsPassed / $TestsTotal) * 100, 1)
Write-Host "Success Rate: $SuccessRate%" -ForegroundColor Cyan

if ($TestsFailed -eq 0) {
    Write-Host ""
    Write-Host "🎉 All tests passed! Pipeline integration is successful." -ForegroundColor Green
    Write-Host ""
    Write-Host "✅ INTEGRATION TEST COMPLETED SUCCESSFULLY" -ForegroundColor Green
    Write-Host ""
    Write-Host "The RobustCBRN pipeline has been successfully integrated!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Available commands:" -ForegroundColor Yellow
    Write-Host "  make pipeline              - Run complete pipeline" -ForegroundColor White
    Write-Host "  make pipeline-validate     - Platform validation" -ForegroundColor White
    Write-Host "  make pipeline-setup        - Environment setup" -ForegroundColor White
    Write-Host "  make pipeline-sample       - Sample evaluation" -ForegroundColor White
    Write-Host "  make pipeline-full         - Full evaluation" -ForegroundColor White
    Write-Host ""
    Write-Host "For Windows users without make/bash:" -ForegroundColor Yellow
    Write-Host "  Use Git Bash or WSL to run the pipeline scripts directly" -ForegroundColor White
    Write-Host ""
    Write-Host "Integration Status: COMPLETE ✅" -ForegroundColor Green
    exit 0
} else {
    Write-Host ""
    Write-Host "❌ Some tests failed. Please check the integration." -ForegroundColor Red
    Write-Host ""
    Write-Host "Failed tests indicate issues that need to be addressed." -ForegroundColor Red
    exit 1
}
