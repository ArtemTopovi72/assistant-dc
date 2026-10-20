# One-time, as administrator: the PawnIO driver LibreHardwareMonitor reads the CPU through, and a
# logon task that runs scripts/sensors_service.py elevated (the admin panel's CPU temperature and power).
#   powershell -ExecutionPolicy Bypass -File scripts\install_sensors.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Start-Process powershell -Verb RunAs -ArgumentList "-ExecutionPolicy Bypass -File `"$PSCommandPath`""
    exit
}
winget install --id namazso.PawnIO -e --accept-source-agreements --accept-package-agreements
$py = Join-Path $root "venv\Scripts\pythonw.exe"
$action = New-ScheduledTaskAction -Execute $py -Argument "`"$root\scripts\sensors_service.py`"" -WorkingDirectory $root
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -RunLevel Highest -LogonType Interactive
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit 0 -RestartCount 99 -RestartInterval (New-TimeSpan -Minutes 1)
Register-ScheduledTask -TaskName "AssistantSensors" -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Force | Out-Null
Start-ScheduledTask -TaskName "AssistantSensors"
Write-Host "AssistantSensors started: runtime\sensors.json"
