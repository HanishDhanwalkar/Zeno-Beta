param(
    [string]$Distribution = "Ubuntu",
    [int]$Port = 6380,
    [int]$StartupTimeoutSeconds = 30
)

$ErrorActionPreference = "Stop"

function Test-RedisPing {
    $client = [System.Net.Sockets.TcpClient]::new()
    try {
        $client.Connect("127.0.0.1", $Port)
        $stream = $client.GetStream()
        $stream.ReadTimeout = 2000
        $command = [System.Text.Encoding]::ASCII.GetBytes("*1`r`n`$4`r`nPING`r`n")
        $stream.Write($command, 0, $command.Length)
        $reader = [System.IO.StreamReader]::new(
            $stream,
            [System.Text.Encoding]::ASCII,
            $false,
            1024,
            $true
        )
        return $reader.ReadLine() -eq "+PONG"
    }
    catch {
        return $false
    }
    finally {
        $client.Dispose()
    }
}

function Test-RedisVectorSearch {
    $suffix = [Guid]::NewGuid().ToString("N")
    $index = "zeno:startup-check:$suffix"
    $prefix = "zeno:startup-check-record:$suffix`:"

    $createOutput = & wsl.exe --distribution $Distribution --user root --exec redis-cli `
        -p $Port FT.CREATE $index ON HASH PREFIX 1 $prefix SCHEMA vector VECTOR HNSW 6 `
        TYPE FLOAT32 DIM 2 DISTANCE_METRIC COSINE 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Redis Search vector-index check failed: $($createOutput -join ' ')"
    }

    $dropOutput = & wsl.exe --distribution $Distribution --user root --exec redis-cli `
        -p $Port FT.DROPINDEX $index 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Could not remove the temporary Redis Search index: $($dropOutput -join ' ')"
    }
}

if (-not (Test-RedisPing)) {
    $stdoutLog = Join-Path $env:TEMP "zeno-redis-wsl.stdout.log"
    $stderrLog = Join-Path $env:TEMP "zeno-redis-wsl.stderr.log"
    $arguments = @(
        "--distribution", $Distribution,
        "--user", "root",
        "--exec", "redis-server",
        "/etc/redis/redis.conf",
        "--daemonize", "no"
    )

    $process = Start-Process -FilePath "wsl.exe" `
        -ArgumentList $arguments `
        -WindowStyle Hidden `
        -PassThru `
        -RedirectStandardOutput $stdoutLog `
        -RedirectStandardError $stderrLog

    $deadline = (Get-Date).AddSeconds($StartupTimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if (Test-RedisPing) {
            break
        }

        if ($process.HasExited) {
            $details = if (Test-Path $stderrLog) {
                Get-Content $stderrLog -Raw
            }
            else {
                "No WSL error output was captured."
            }
            throw "Redis did not start in WSL. $details"
        }

        Start-Sleep -Seconds 1
    }
}

if (-not (Test-RedisPing)) {
    throw "Redis did not respond at localhost:$Port within $StartupTimeoutSeconds seconds."
}

Test-RedisVectorSearch
Write-Host "Redis with vector Search is ready at redis://localhost:$Port/0 (WSL: $Distribution)."
