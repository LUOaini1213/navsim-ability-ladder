# Split remaining PrivMap→Human gap: geometry vs speed.
$ErrorActionPreference = "Stop"
$root = if ($env:NAVSIM_WORKSPACE) { $env:NAVSIM_WORKSPACE } else { (Get-Location).Path }
$py = if ($env:NAVSIM_PYTHON) { $env:NAVSIM_PYTHON } else { "python" }
$env:SSL_CERT_FILE = & $py -c "import certifi; print(certifi.where())"
$env:REQUESTS_CA_BUNDLE = $env:SSL_CERT_FILE

$jobs = @(
  @{ agent = "privileged_centerline_kin"; name = "privileged_centerline_kin" },
  @{ agent = "privileged_centerline_gtspeed"; name = "privileged_centerline_gtspeed" },
  @{ agent = "privileged_gtpath_kin"; name = "privileged_gtpath_kin" }
)
foreach ($j in $jobs) {
  Write-Host "==== PDM $($j.agent) ===="
  & $py "$root\run_navsim_step.py" "$root\navsim\navsim\planning\script\run_pdm_score.py" `
    train_test_split=warmup_test_e2e `
    worker=sequential `
    "agent=$($j.agent)" `
    "experiment_name=$($j.name)" `
    metric_cache_path=$env:NAVSIM_EXP_ROOT/metric_cache
  if ($LASTEXITCODE -ne 0) { throw "pdm failed for $($j.agent)" }
}
