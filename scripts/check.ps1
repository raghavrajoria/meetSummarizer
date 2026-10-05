param([string]$Python = "python")
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
& $Python -m ruff check backend indicmeet tests scripts
if ($LASTEXITCODE) { exit $LASTEXITCODE }
& $Python -m pytest -q --tb=short
if ($LASTEXITCODE) { exit $LASTEXITCODE }
& $Python scripts/validate_contract.py fixtures
if ($LASTEXITCODE) { exit $LASTEXITCODE }
npm --prefix frontend ci
if ($LASTEXITCODE) { exit $LASTEXITCODE }
npm --prefix frontend run build
exit $LASTEXITCODE
