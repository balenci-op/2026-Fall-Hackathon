<#
Double-click entry point for the local candidate review application.
Only starts the existing project Python; never installs packages or downloads data.
Optional Port, DataDir, NoBrowser, and NoDialog parameters support isolated checks.
#>
[CmdletBinding()]
param(
    [ValidateSet('Launch', 'Stop')]
    [string]$Action = 'Launch',
    [ValidateRange(1024, 65535)]
    [int]$Port = 8765,
    [string]$DataDir = '',
    [switch]$NoBrowser,
    [switch]$NoDialog
)

$ErrorActionPreference = 'Stop'
$taskProject = $PSScriptRoot
$taskBaseUrl = "http://127.0.0.1:$Port"
$taskAppName = 'ai4infra-candidate-review'

function Get-ReviewHealth {
    try {
        $taskHealth = Invoke-RestMethod -Uri "$taskBaseUrl/health" -TimeoutSec 2
        if ($taskHealth.app -eq $taskAppName -and $taskHealth.ready -eq $true) {
            return $taskHealth
        }
    }
    catch { }
    return $null
}

function Test-LocalPort {
    $taskClient = New-Object System.Net.Sockets.TcpClient
    try {
        $taskConnect = $taskClient.ConnectAsync('127.0.0.1', $Port)
        return ($taskConnect.Wait(700) -and $taskClient.Connected)
    }
    catch { return $false }
    finally { $taskClient.Dispose() }
}

function Show-LauncherMessage([string]$Message, [bool]$IsError = $false) {
    if ($NoDialog) {
        Write-Output $Message
        return
    }
    Add-Type -AssemblyName System.Windows.Forms
    $taskIcon = if ($IsError) {
        [System.Windows.Forms.MessageBoxIcon]::Error
    } else {
        [System.Windows.Forms.MessageBoxIcon]::Information
    }
    [void][System.Windows.Forms.MessageBox]::Show(
        $Message, 'Candidate Review', [System.Windows.Forms.MessageBoxButtons]::OK, $taskIcon
    )
}

function Open-ReviewBrowser {
    if (-not $NoBrowser) {
        # The browser is the requested visible user interface.
        Start-Process -FilePath "$taskBaseUrl/"
    }
}

$taskMutex = $null
$taskOwnsMutex = $false
try {
    # Serialize double-clicks so two launchers cannot start competing servers.
    $taskMutex = New-Object System.Threading.Mutex($false, "Local\AI4InfraCandidateReview$Port")
    try { $taskOwnsMutex = $taskMutex.WaitOne(30000) }
    catch [System.Threading.AbandonedMutexException] { $taskOwnsMutex = $true }
    if (-not $taskOwnsMutex) {
        throw 'Another Candidate Review launcher is still starting. Try again in a moment.'
    }

    $taskHealth = Get-ReviewHealth
    if ($Action -eq 'Stop') {
        if ($null -eq $taskHealth) {
            if (Test-LocalPort) {
                throw "Port $Port is occupied by an unrecognized service. Nothing was stopped."
            }
            Show-LauncherMessage 'Candidate Review is already stopped.'
            exit 0
        }
        $taskState = Invoke-RestMethod -Uri "$taskBaseUrl/api/state" -TimeoutSec 5
        if (-not $taskState.session_token) {
            throw 'The review service did not provide its shutdown token. Nothing was stopped.'
        }
        [void](Invoke-RestMethod -Method Post -Uri "$taskBaseUrl/api/shutdown" -TimeoutSec 5 `
            -Headers @{ 'X-Review-Token' = $taskState.session_token; 'Origin' = $taskBaseUrl } `
            -ContentType 'application/json' -Body '{}')
        $taskStopTimer = [System.Diagnostics.Stopwatch]::StartNew()
        while ((Test-LocalPort) -and $taskStopTimer.Elapsed.TotalSeconds -lt 10) {
            Start-Sleep -Milliseconds 200
        }
        if (Test-LocalPort) {
            throw 'The server accepted shutdown but is still closing. Try Stop again in a moment.'
        }
        Show-LauncherMessage 'Candidate Review stopped. Saved reviews are on disk.'
        exit 0
    }

    if ($null -ne $taskHealth) {
        Open-ReviewBrowser
        Write-Output "Candidate Review is already running at $taskBaseUrl/"
        exit 0
    }
    if (Test-LocalPort) {
        throw "Port $Port is occupied but does not identify itself as Candidate Review. Close that service or use a different port."
    }

    $taskPython = Join-Path $taskProject '.venv-ground\Scripts\python.exe'
    $taskApp = Join-Path $taskProject 'review_app.py'
    if (-not (Test-Path -LiteralPath $taskPython -PathType Leaf)) {
        throw "The project Python environment is missing: $taskPython`nRestore .venv-ground before launching. No packages were installed."
    }
    if (-not (Test-Path -LiteralPath $taskApp -PathType Leaf)) {
        throw "The review application is missing: $taskApp"
    }
    $taskOutput = if ($DataDir) {
        [System.IO.Path]::GetFullPath($DataDir)
    } else {
        Join-Path $taskProject 'outputs\objects\reference_set'
    }
    [void](New-Item -ItemType Directory -Path $taskOutput -Force)
    $taskLogStamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
    $taskLog = Join-Path $taskOutput "server_$taskLogStamp.log"
    $taskErrorLog = Join-Path $taskOutput "server_$taskLogStamp.error.log"
    $taskArguments = @('-X', 'utf8', '-u', ('"' + $taskApp + '"'), '--port', "$Port", '--no-browser')
    if ($DataDir) {
        # A trailing backslash must be doubled before a closing Windows quote.
        $taskQuotedOutput = '"' + ($taskOutput -replace '(\\+)$', '$1$1') + '"'
        $taskArguments += @('--data-dir', $taskQuotedOutput)
    }
    $taskProcess = Start-Process -FilePath $taskPython -ArgumentList $taskArguments `
        -WorkingDirectory $taskProject -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput $taskLog -RedirectStandardError $taskErrorLog

    $taskStartupTimer = [System.Diagnostics.Stopwatch]::StartNew()
    do {
        Start-Sleep -Milliseconds 250
        $taskHealth = Get-ReviewHealth
        if ($null -ne $taskHealth) {
            Open-ReviewBrowser
            Write-Output "Candidate Review ready at $taskBaseUrl/"
            exit 0
        }
        $taskProcess.Refresh()
        if ($taskProcess.HasExited) {
            throw "Candidate Review exited during startup.`nDetails: $taskErrorLog"
        }
    } while ($taskStartupTimer.Elapsed.TotalSeconds -lt 45)
    throw "Candidate Review has not become ready yet. It may still be starting; try Launch again shortly.`nDetails: $taskLog`nErrors: $taskErrorLog"
}
catch {
    Show-LauncherMessage $_.Exception.Message $true
    exit 1
}
finally {
    if ($taskOwnsMutex -and $null -ne $taskMutex) { $taskMutex.ReleaseMutex() }
    if ($null -ne $taskMutex) { $taskMutex.Dispose() }
}
