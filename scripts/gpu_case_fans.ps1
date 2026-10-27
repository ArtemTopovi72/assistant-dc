param([switch]$SelfTest)
$ErrorActionPreference = 'Stop'

function Get-FanTarget($Core, $Memory) {
    if ($null -eq $Core -or $null -eq $Memory -or $Core -lt 0 -or $Core -gt 125 -or $Memory -lt 0 -or $Memory -gt 125) { return 100 }
    $gpu = 50 + ($Core - 40) * 50 / 35
    $vram = 50 + ($Memory - 60) * 50 / 25
    return [math]::Min(100, [math]::Max(50, [math]::Max($gpu, $vram)))
}

if ($SelfTest) {
    foreach ($case in @(@(35, 50, 50), @(75, 60, 100), @(40, 85, 100), @($null, 50, 100), @(40, $null, 100), @(200, 50, 100))) {
        if ((Get-FanTarget $case[0] $case[1]) -ne $case[2]) { throw 'Fan curve self-test failed' }
    }
    'Fan curve self-test passed'
    exit 0
}

$dir = Join-Path $env:LOCALAPPDATA 'AssistantDC\Cooling'
$status = Join-Path $dir 'status.json'
$stop = Join-Path $dir 'stop'
Add-Type -Path (Join-Path $dir 'monitor\LibreHardwareMonitorLib.dll')
$computer = New-Object LibreHardwareMonitor.Hardware.Computer
$computer.IsMotherboardEnabled = $true
$computer.IsGpuEnabled = $true
$controls = @()
$percent = 100
try {
    $computer.Open()
    $board = @($computer.Hardware | ForEach-Object { $_.SubHardware } | Where-Object { [string]$_.Identifier -eq '/lpc/nct6796dr/0' })
    $gpu = @($computer.Hardware | Where-Object { [string]$_.Identifier -eq '/gpu-nvidia/0' })
    if ($board.Count -ne 1) { throw 'Expected NCT6796D-R controller not found' }
    $board = $board[0]
    $board.Update()
    # Only physical channels 3-6: radiator (1), pump (2) and unused output (7) stay with BIOS.
    $controls = @(foreach ($i in 2..5) {
        $s = @($board.Sensors | Where-Object { [string]$_.Identifier -eq "/lpc/nct6796dr/0/control/$i" })
        if ($s.Count -ne 1 -or !$s[0].Control) { throw "Case fan control $i not found" }
        $s[0].Control
    })
    while (!(Test-Path -LiteralPath $stop)) {
        $core = $null; $memory = $null; $errorText = $null
        try {
            if ($gpu.Count -eq 1) {
                $gpu[0].Update()
                $core = ($gpu[0].Sensors | Where-Object { [string]$_.Identifier -eq '/gpu-nvidia/0/temperature/0' }).Value
                $memory = ($gpu[0].Sensors | Where-Object { [string]$_.Identifier -eq '/gpu-nvidia/0/temperature/3' }).Value
            }
            $target = Get-FanTarget $core $memory
            # Respond immediately to heat, decay slowly after a render ends.
            $percent = [math]::Max($target, $percent - 2)
            foreach ($control in $controls) { $control.SetSoftware([single]$percent) }
            $board.Update()
        } catch {
            $errorText = $_.Exception.Message
            $percent = 100
            foreach ($control in $controls) { try { $control.SetSoftware(100) } catch {} }
        }
        $data = @{ time = (Get-Date).ToString('o'); core = $core; memory = $memory; percent = $percent; error = $errorText
            fans = @($board.Sensors | Where-Object { [string]$_.SensorType -eq 'Fan' } | ForEach-Object { @{ name = $_.Name; rpm = $_.Value } })
            pwm = @($board.Sensors | Where-Object { [string]$_.SensorType -eq 'Control' } | ForEach-Object { @{ name = $_.Name; percent = $_.Value } }) }
        try {
            $data | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath "$status.tmp" -Encoding UTF8
            Move-Item -LiteralPath "$status.tmp" -Destination $status -Force
        } catch { Write-Warning "Cooling status could not be saved: $_" }
        Start-Sleep -Seconds 2
    }
} finally {
    foreach ($control in $controls) { try { $control.SetDefault() } catch {} }
    $computer.Close()
}
