param([string]$Python = "python")
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
& $Python -m ruff check backend indicmeet asr_service tests scripts
if ($LASTEXITCODE) { exit $LASTEXITCODE }
& $Python -m pytest -q --tb=short
if ($LASTEXITCODE) { exit $LASTEXITCODE }
& $Python scripts/validate_contract.py fixtures
if ($LASTEXITCODE) { exit $LASTEXITCODE }
Push-Location frontend
try {
    npm ci
    if ($LASTEXITCODE) { exit $LASTEXITCODE }
    npm run build
    exit $LASTEXITCODE
} finally { Pop-Location }
