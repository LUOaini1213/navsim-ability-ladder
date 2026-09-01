# Train the learned map-aware MLP (fills the ladder gap).
$ErrorActionPreference = "Stop"
$root = if ($env:NAVSIM_WORKSPACE) { $env:NAVSIM_WORKSPACE } else { (Get-Location).Path }
$py = if ($env:NAVSIM_PYTHON) { $env:NAVSIM_PYTHON } else { "python" }
& $py "$root\run_navsim_step.py" "$root\train_map_mlp.py" --epochs 80
if ($LASTEXITCODE -ne 0) { throw "map mlp training failed" }
