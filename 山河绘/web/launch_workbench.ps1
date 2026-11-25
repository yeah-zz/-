$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$baseUrl = 'http://127.0.0.1:8765'
$pythonPath = 'D:\Anaconda3\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    $pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $pythonPath)) { $pythonPath = 'python' }
}
Set-Location -LiteralPath $projectRoot
$env:PYTHONIOENCODING = 'utf-8'

function Open-Existing {
    param([string]$Message)
    Write-Host $Message
    Start-Process $baseUrl
}

function Is-Busy {
    $state = Invoke-RestMethod "$baseUrl/api/state" -TimeoutSec 5
    $drafts = Invoke-RestMethod "$baseUrl/api/studio/jobs" -TimeoutSec 5
    $library = Invoke-RestMethod "$baseUrl/api/library" -TimeoutSec 5
    return ($state.parametersLocked -or
        $state.status -notin @('idle','complete','stopped','saved','error') -or
        @($drafts.jobs | Where-Object { $_.busy }).Count -gt 0 -or
        @($library.operations | Where-Object { $_.status -eq 'building' }).Count -gt 0 -or
        (Test-Path -LiteralPath (Join-Path $projectRoot 'robot_runs/painting_session.lock')) -or
        (Test-Path -LiteralPath (Join-Path $projectRoot 'robot_runs/blue_wash_motion.lock')))
}

try {
    $listener = @(Get-NetTCPConnection -LocalPort 8765 -State Listen -ErrorAction SilentlyContinue)
    if ($listener.Count -gt 0) {
        $config = Invoke-RestMethod "$baseUrl/api/config" -TimeoutSec 5
        if (-not $config.token -or -not $config.labels.black) { throw 'Port 8765 belongs to another service.' }
        if ($config.root -and [IO.Path]::GetFullPath($config.root) -ne [IO.Path]::GetFullPath($projectRoot)) {
            throw 'Another project is using port 8765. No process was stopped.'
        }
        if (Is-Busy) {
            Open-Existing 'A painting or image task is active. Opened the page without restarting.'
            exit 0
        }
        if ($config.serviceRestart) {
            try {
                Invoke-RestMethod "$baseUrl/api/service/restart" -Method Post -ContentType 'application/json' `
                    -Headers @{'X-Workbench-Token'=$config.token} -Body '{}' -TimeoutSec 10 | Out-Null
            } catch {
                Open-Existing 'The service could not restart safely. Opened the existing page.'
                exit 0
            }
        } else {
            # One-time compatibility with the original workbench. Target only
            # the verified localhost listener, never other Python processes.
            $ownerIds = @($listener.OwningProcess | Sort-Object -Unique)
            if ($ownerIds.Count -ne 1) { throw 'Cannot identify the workbench process safely.' }
            $serviceProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $($ownerIds[0])"
            $expectedScript = Join-Path $projectRoot 'web/server.py'
            $relativeCommand = '^"?[^"\r\n]*python(?:\.exe)?"?\s+(?:-u\s+)?web[\\/]server\.py(?:\s+--open-browser)?\s*$'
            if (-not $serviceProcess -or ($serviceProcess.CommandLine -notmatch $relativeCommand -and
                -not $serviceProcess.CommandLine.Contains($expectedScript))) {
                throw 'The process is not the expected workbench. No process was stopped.'
            }
            if (Is-Busy) {
                Open-Existing 'A task started. Opened the page without restarting.'
                exit 0
            }
            Stop-Process -Id $serviceProcess.ProcessId -ErrorAction Stop
        }
        $deadline = [DateTime]::UtcNow.AddSeconds(12)
        do {
            $remaining = @(Get-NetTCPConnection -LocalPort 8765 -State Listen -ErrorAction SilentlyContinue)
            if ($remaining.Count -eq 0) { break }
            Start-Sleep -Milliseconds 200
        } while ([DateTime]::UtcNow -lt $deadline)
        if ($remaining.Count -gt 0) { throw 'The old service has not exited. A second service was not started.' }
    }
    # Probe the selected environment without installing or changing any packages.
    & $pythonPath -c 'import sys; assert sys.version_info >= (3, 10); import numpy, cv2'
    if ($LASTEXITCODE -ne 0) {
        throw "Python or dependencies are unavailable: $pythonPath. The existing environment was not modified."
    }
    Write-Host ('Using Python: ' + $pythonPath)
    Write-Host 'Starting the latest workbench. Close this window or press Ctrl+C to stop the web service.'
    & $pythonPath -u (Join-Path $projectRoot 'web/server.py') --open-browser
    exit $LASTEXITCODE
} catch {
    Write-Host ('Workbench startup failed: ' + $_.Exception.Message)
    exit 1
}
