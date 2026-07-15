$ErrorActionPreference = "Stop"

$root = Resolve-Path (Join-Path $PSScriptRoot "..")
$python = "C:\Users\vinee\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
$deps = Resolve-Path (Join-Path $root "backend\.deps")

$env:PYTHONDONTWRITEBYTECODE = "1"
$env:PYTHONPATH = "$deps;$root"

Start-Process `
  -FilePath $python `
  -ArgumentList @("-B", "-m", "uvicorn", "backend.app.main:app", "--host", "127.0.0.1", "--port", "8000") `
  -WorkingDirectory $root `
  -WindowStyle Hidden
