<#
.SYNOPSIS
  Spike 0's GPU row: what Fritz costs the GPU, animating against parked.

.DESCRIPTION
  Run while Mister Fritz is running and on screen, on an otherwise idle
  desktop. Samples Windows' "GPU Engine" counters (what Task Manager reads)
  for -Seconds with Fritz animating, presses Ctrl+Alt+H to park him, samples
  again, then presses Ctrl+Alt+H to bring him back.

  It reports, per GPU adapter, the total 3D-engine use of every process in
  both states and the difference — the honest price of Fritz, since the
  desktop compositor (dwm.exe) does part of the work of drawing him and that
  is billed to dwm, not to Fritz — plus dwm's and Fritz's WebView2 GPU
  process's own shares, and which adapter Fritz renders on.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File desktop\scripts\spike0-gpu.ps1 -Seconds 60
#>
param([int]$Seconds = 60)
$ErrorActionPreference = 'Stop'

Add-Type @"
using System;
using System.Runtime.InteropServices;
public static class SpikeKeys {
  [DllImport("user32.dll")] public static extern void keybd_event(byte vk, byte scan, uint flags, UIntPtr extra);
  public static void CtrlAlt(byte key) {
    keybd_event(0x11,0,0,UIntPtr.Zero); keybd_event(0x12,0,0,UIntPtr.Zero);
    keybd_event(key,0,0,UIntPtr.Zero); System.Threading.Thread.Sleep(60);
    keybd_event(key,0,2,UIntPtr.Zero); keybd_event(0x12,0,2,UIntPtr.Zero); keybd_event(0x11,0,2,UIntPtr.Zero);
  }
}
"@

$gpuProc = Get-CimInstance Win32_Process -Filter "Name='msedgewebview2.exe'" |
  Where-Object { $_.CommandLine -match 'webview-exe-name=mister-fritz-desktop.exe' -and $_.CommandLine -match '--type=gpu-process' } |
  Select-Object -First 1
if (-not $gpuProc) { throw "Mister Fritz isn't running (no WebView2 GPU process for mister-fritz-desktop.exe)." }
$fritz = $gpuProc.ProcessId
# The compositor in Fritz's own session (fast user switching or RDP adds more).
$dwm = (Get-Process dwm | Where-Object SessionId -eq $gpuProc.SessionId | Select-Object -First 1).Id
if (-not $dwm) { throw "No dwm.exe in Fritz's session." }

function Sample([string]$label) {
  Write-Host "Sampling $Seconds s: $label..."
  # A process that exits mid-sample makes Get-Counter report an invalid
  # sample; that must not end the run (Fritz would be left parked).
  $c = Get-Counter '\GPU Engine(*engtype_3D)\Utilization Percentage' -SampleInterval 1 -MaxSamples $Seconds -ErrorAction SilentlyContinue
  $rows = foreach ($set in $c.CounterSamples) {
    if ($set.Status -eq 0 -and $set.InstanceName -match '^pid_(\d+)_luid_(0x[0-9a-f]+_0x[0-9a-f]+)_') {
      [pscustomobject]@{ Pid = [int]$Matches[1]; Luid = $Matches[2]; Value = $set.CookedValue }
    }
  }
  # Per adapter: sum over processes and engines per sample, then the mean over samples.
  $byLuid = @{}
  foreach ($g in ($rows | Group-Object Luid)) {
    $byLuid[$g.Name] = [pscustomobject]@{
      Total = ($g.Group | Measure-Object Value -Sum).Sum / $Seconds
      Dwm   = (($g.Group | Where-Object Pid -eq $dwm) | Measure-Object Value -Sum).Sum / $Seconds
      Fritz = (($g.Group | Where-Object Pid -eq $fritz) | Measure-Object Value -Sum).Sum / $Seconds
    }
  }
  $byLuid
}

$animating = Sample 'Fritz animating'
[SpikeKeys]::CtrlAlt(0x48)   # Ctrl+Alt+H: park
try {
  Start-Sleep -Seconds 2
  $parked = Sample 'Fritz parked'
} finally {
  [SpikeKeys]::CtrlAlt(0x48) # and back, whatever happened (an error, Ctrl+C)
}

$fritzAdapter = ($animating.GetEnumerator() | Sort-Object { $_.Value.Fritz } -Descending | Select-Object -First 1).Key
Write-Host ""
Write-Host "3D engine use, mean over $Seconds s (percent of one adapter):"
foreach ($luid in ($animating.Keys + $parked.Keys | Sort-Object -Unique)) {
  $a = $animating[$luid]; $p = $parked[$luid]
  $mark = if ($luid -eq $fritzAdapter) { '   <- Fritz renders here' } else { '' }
  "{0}{1}" -f $luid, $mark
  "   total   animating {0,6:N2}   parked {1,6:N2}   difference {2,6:N2}" -f $a.Total, $p.Total, ($a.Total - $p.Total)
  "   dwm     animating {0,6:N2}   parked {1,6:N2}" -f $a.Dwm, $p.Dwm
  "   Fritz   animating {0,6:N2}   parked {1,6:N2}" -f $a.Fritz, $p.Fritz
}
Write-Host ""
Write-Host "Record the 'difference' on Fritz's adapter in SPIKE0.md's GPU row. Task Manager's"
Write-Host "Performance tab names the adapters; the one whose 3D graph moved is Fritz's."
