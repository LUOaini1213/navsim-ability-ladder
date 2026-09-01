# Blind kinematic baseline (accel + command yaw) on warmup_test_e2e.
$ErrorActionPreference = "Stop"
$root = if ($env:NAVSIM_WORKSPACE) { $env:NAVSIM_WORKSPACE } else { (Get-Location).Path }
$py = if ($env:NAVSIM_PYTHON) { $env:NAVSIM_PYTHON } else { "python" }
$env:SSL_CERT_FILE = & $py -c "import certifi; print(certifi.where())"
$env:REQUESTS_CA_BUNDLE = $env:SSL_CERT_FILE

Write-Host "==== PDM score KinematicAgent warmup_test_e2e ===="
& $py "$root\run_navsim_step.py" "$root\navsim\navsim\planning\script\run_pdm_score.py" `
  train_test_split=warmup_test_e2e `
  worker=sequential `
  agent=kinematic_agent `
  experiment_name=kinematic_agent_mini `
  metric_cache_path=$env:NAVSIM_EXP_ROOT/metric_cache
if ($LASTEXITCODE -ne 0) { throw "kinematic pdm score failed" }
