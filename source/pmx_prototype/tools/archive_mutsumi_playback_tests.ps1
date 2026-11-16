# Archive only the identified playback experiments; never delete model/training assets.
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$outputRoot = Join-Path $projectRoot 'output'
$archiveRoot = Join-Path $outputRoot ('archive\mutsumi_playback_history_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
if (Test-Path -LiteralPath $archiveRoot) { throw 'Archive already exists' }
$candidates = [System.Collections.Generic.List[string]]::new()
foreach ($name in @('mutsumi_dance_realtime_optimization','mutsumi_dance_perf_v2','mutsumi_dance_perf_v3')) {
    $path = Join-Path $outputRoot $name
    if (Test-Path -LiteralPath $path) { $candidates.Add($path) }
}
$acceptedRoot = Join-Path $outputRoot 'mutsumi_startup_v4'
if (Test-Path -LiteralPath $acceptedRoot) {
    foreach ($entry in Get-ChildItem -LiteralPath $acceptedRoot -Force) {
        if ($entry.PSIsContainer -and $entry.Name -eq 'final') { continue }
        if (-not $entry.PSIsContainer -and $entry.Name -in @('deepbluetown_prewarm_before_after_2k.mp4','initial_pose_detail_before_after.mp4','verification.json','README.txt')) { continue }
        $candidates.Add($entry.FullName)
    }
}
foreach ($file in Get-ChildItem -LiteralPath $outputRoot -File) {
    if ($file.Name -match '^mutsumi_(dance_|perf_|settle_|startup_v4_).*\.log$' -or $file.Name -in @(
        'benchmark_mutsumi_dance_threads.py','compare_mutsumi_settle.py','finalize_mutsumi_startup.py',
        'setup_mutsumi_comparison_scene.py','test_mutsumi_startup.py','run_mutsumi_v3_stability.py',
        'maintain_audio_library.py','record_mutsumi_v10_library.py','verify_mutsumi_v10_library.py')) {
        $candidates.Add($file.FullName)
    }
}
$activeCommands = @(Get-CimInstance Win32_Process | Where-Object { $_.Name -in @('kit.exe','python.exe','pythonw.exe') } | Select-Object -ExpandProperty CommandLine)
$items = @(); $skipped = @()
foreach ($candidate in $candidates) {
    $sourcePath = (Resolve-Path -LiteralPath $candidate).Path
    if (-not $sourcePath.StartsWith($outputRoot + '\',[StringComparison]::OrdinalIgnoreCase)) { throw "Source escaped output: $sourcePath" }
    $relative = [IO.Path]::GetRelativePath($outputRoot,$sourcePath)
    $destination = [IO.Path]::GetFullPath((Join-Path $archiveRoot $relative))
    if (-not $destination.StartsWith($archiveRoot + '\',[StringComparison]::OrdinalIgnoreCase)) { throw 'Archive path escaped target' }
    $needle = ('output/' + $relative.Replace('\','/')).ToLowerInvariant()
    if (@($activeCommands | Where-Object { $_ -and $_.Replace('\','/').ToLowerInvariant().Contains($needle) }).Count) {
        $skipped += $relative; continue
    }
    $files = @(); $entry = Get-Item -LiteralPath $sourcePath
    $sourceFiles = if ($entry.PSIsContainer) { @(Get-ChildItem -LiteralPath $sourcePath -File -Recurse -Force) } else { @($entry) }
    foreach ($file in $sourceFiles) {
        $fileRelative = [IO.Path]::GetRelativePath($outputRoot,$file.FullName)
        $files += [ordered]@{original=$file.FullName; archived=(Join-Path $archiveRoot $fileRelative); bytes=$file.Length; sha256=(Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()}
    }
    $items += [ordered]@{source=$sourcePath; destination=$destination; status='planned'; files=$files}
}
New-Item -ItemType Directory -Path $archiveRoot -Force | Out-Null
$manifest = [ordered]@{status='planned'; project=$projectRoot; created=(Get-Date -Format o); skipped_active=$skipped; items=$items}
$manifestPath = Join-Path $archiveRoot 'manifest.json'
$manifest | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $manifestPath -Encoding utf8
foreach ($item in $items) {
    if (Test-Path -LiteralPath $item.destination) { throw "Refusing to overwrite $($item.destination)" }
    New-Item -ItemType Directory -Path (Split-Path -Parent $item.destination) -Force | Out-Null
    Move-Item -LiteralPath $item.source -Destination $item.destination
    foreach ($file in $item.files) {
        if ((Get-FileHash -LiteralPath $file.archived -Algorithm SHA256).Hash.ToLowerInvariant() -ne $file.sha256) { throw "Archive hash mismatch: $($file.archived)" }
    }
    $item.status='archived'
}
$manifest.status='complete'
$manifest | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $manifestPath -Encoding utf8
$fileCount=0; $totalBytes=0L
foreach ($item in $items) { foreach ($file in $item.files) { $fileCount++; $totalBytes += [long]$file.bytes } }
[ordered]@{archive=$archiveRoot; items=$items.Count; files=$fileCount; bytes=$totalBytes; skipped_active=$skipped} | ConvertTo-Json -Depth 4
