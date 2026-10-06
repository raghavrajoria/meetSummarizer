param(
    [Parameter(Mandatory=$true)][string]$Source,
    [double]$Start = 840,
    [double]$Duration = 300,
    [string]$DiarizationCsv,
    [string]$Out = 'data/local-real/agm-300.wav',
    [string]$Python = '.venv/Scripts/python.exe'
)
$ErrorActionPreference = 'Stop'
if ($Start -lt 0 -or $Duration -le 0) { throw 'Invalid clip interval' }
if (Test-Path -LiteralPath $Out) { throw 'Refusing to overwrite clip' }
New-Item -ItemType Directory -Force -Path (Split-Path -Parent ([IO.Path]::GetFullPath($Out))) | Out-Null
& ffmpeg -nostdin -v error -ss $Start -i $Source -t $Duration -vn -ar 16000 -ac 1 -c:a pcm_s16le -n $Out
if ($LASTEXITCODE -ne 0) { throw 'ffmpeg clip failed' }
$clipArgs = @('scripts/clip_turns.py', '--audio', $Out, '--start', "$Start", '--duration', "$Duration")
if ($DiarizationCsv) { $clipArgs += @('--csv', $DiarizationCsv) }
& $Python @clipArgs
if ($LASTEXITCODE -ne 0) { throw 'turn generation failed' }
