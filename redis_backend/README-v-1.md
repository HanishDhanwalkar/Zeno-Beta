# Starting Redis in Ubuntu WSL

The v-6 demo needs Redis with the Search / Query Engine vector commands. The
Ubuntu WSL setup uses the official Redis package and listens on port `6380`,
leaving Kali's Redis on `6379` untouched.

## Start Redis

From the repository root, run this in PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\local_tests\redis_backend\start-redis-wsl.ps1
```

The script starts Redis in a hidden WSL process and keeps Ubuntu running so
Windows can reach its localhost port forwarding. It checks both `PING` and
creation of a temporary HNSW vector index before reporting success. Running it
again while Redis is already available is safe.

Configure the demo to use this instance in the same PowerShell window:

```powershell
$env:REDIS_URL = "redis://localhost:6380/0"
```

Then run the v-6 demo as described in [`../README.md`](../README.md).

## Stop Redis

To shut down this Redis instance and allow Ubuntu WSL to stop:

```powershell
wsl.exe -d Ubuntu -- redis-cli -p 6380 shutdown
```

Run the start script again after a Windows restart, WSL shutdown, or if the
Redis connection is refused. If startup fails, check that Ubuntu is installed
as the WSL distribution named `Ubuntu` and inspect
`%TEMP%\zeno-redis-wsl.stderr.log`.

The Redis server is bound to WSL loopback; do not expose it on a public or
untrusted network.
