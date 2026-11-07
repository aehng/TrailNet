param(
    [Parameter(Mandatory = $true)][string]$Executable,
    [Parameter(Mandatory = $true)][string[]]$ArgumentList,
    [Parameter(Mandatory = $true)][string]$CompletionPath,
    [int]$FallbackSeconds = 300,
    [int]$TimeoutSeconds = 2700
)

$completionDirectory = Split-Path -Parent $CompletionPath
New-Item -ItemType Directory -Force -Path $completionDirectory | Out-Null
$info = [System.Diagnostics.ProcessStartInfo]::new()
$info.FileName = $Executable
$info.UseShellExecute = $false
$info.CreateNoWindow = $false
foreach ($argument in $ArgumentList) {
    [void]$info.ArgumentList.Add($argument)
}
$process = [System.Diagnostics.Process]::new()
$process.StartInfo = $info
$process.EnableRaisingEvents = $true
$sourceIdentifier = "TrailNet.ProcessExit.$([guid]::NewGuid().ToString('N'))"
Register-ObjectEvent -InputObject $process -EventName Exited -SourceIdentifier $sourceIdentifier | Out-Null
$startedAt = [DateTimeOffset]::UtcNow

if (-not $process.Start()) {
    throw "Could not start $Executable"
}

Write-Output "RUN_STARTED pid=$($process.Id) utc=$($startedAt.ToString('o'))"
$finished = $false
$timedOut = $false
while (-not $finished) {
    $elapsed = [DateTimeOffset]::UtcNow - $startedAt
    $event = Wait-Event -SourceIdentifier $sourceIdentifier -Timeout (
        [Math]::Min($FallbackSeconds, [Math]::Max(1, $TimeoutSeconds - $elapsed.TotalSeconds))
    )
    if ($event) {
        Remove-Event -EventIdentifier $event.EventIdentifier
        $finished = $true
        break
    }

    if ($process.HasExited) {
        $finished = $true
        break
    }

    if ($elapsed.TotalSeconds -ge $TimeoutSeconds) {
        Write-Output "RUN_TIMEOUT elapsed_seconds=$([int]$elapsed.TotalSeconds)"
        $timedOut = $true
        $process.Kill($true)
        $process.WaitForExit()
        break
    }

    Write-Output "RUN_FALLBACK elapsed_seconds=$([int]$elapsed.TotalSeconds) still_running=true"
}

$process.Refresh()
$endedAt = [DateTimeOffset]::UtcNow
$result = [ordered]@{
    executable = $Executable
    arguments = $ArgumentList
    pid = $process.Id
    started_at = $startedAt.ToString("o")
    ended_at = $endedAt.ToString("o")
    elapsed_seconds = [int]($endedAt - $startedAt).TotalSeconds
    status = if ($timedOut) { "timeout" } else { "complete" }
    exit_code = if ($process.HasExited) { $process.ExitCode } else { $null }
}
$result | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $CompletionPath -Encoding utf8
Unregister-Event -SourceIdentifier $sourceIdentifier
Write-Output "RUN_CALLBACK status=$($result.status) exit_code=$($result.exit_code) completion=$CompletionPath"
$process.Dispose()
if ($timedOut) { exit 124 }
exit $result.exit_code
