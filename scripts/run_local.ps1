param([int]$Port = 8000)
$ErrorActionPreference = 'Stop'
$projectPath = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectPath '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Root .venv Python is missing.' }
Push-Location $projectPath
$apiProcess = $null
$workerProcess = $null
try {
    & $pythonPath -m alembic upgrade head
    if ($LASTEXITCODE -ne 0) { throw 'Migration failed; API and worker were not started.' }
    $logPath = Join-Path $projectPath 'data\logs'
    New-Item -ItemType Directory -Force -Path $logPath | Out-Null
    $runId = [guid]::NewGuid().ToString('N')
    $apiProcess = Start-Process -FilePath $pythonPath -ArgumentList @('-m', 'uvicorn', 'backend.main:app', '--host', '127.0.0.1', '--port', "$Port", '--no-access-log') -WorkingDirectory $projectPath -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logPath "api-$runId.out.log") -RedirectStandardError (Join-Path $logPath "api-$runId.err.log") -PassThru
    $workerProcess = Start-Process -FilePath $pythonPath -ArgumentList @('-m', 'backend.worker') -WorkingDirectory $projectPath -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logPath "worker-$runId.out.log") -RedirectStandardError (Join-Path $logPath "worker-$runId.err.log") -PassThru
    Write-Host "API: http://127.0.0.1:$Port ; worker started. Logs: $logPath ; Ctrl+C stops both."
    while (-not $apiProcess.HasExited -and -not $workerProcess.HasExited) { Start-Sleep -Seconds 1 }
    throw 'API or worker exited; inspect the process logs.'
} finally {
    foreach ($process in @($apiProcess, $workerProcess)) {
        if ($null -ne $process -and -not $process.HasExited) { Stop-Process -Id $process.Id }
    }
    Pop-Location
}
