param(
    [string]$EnvironmentPath = ".venv-allen-prep",
    [string]$PythonExecutable = ".venv\\Scripts\\python.exe"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$envDir = Join-Path $repoRoot $EnvironmentPath
$requirements = Join-Path $repoRoot "environments\allen_visual_behavior_prep\requirements.txt"
$pythonSource = $PythonExecutable
if (-not [System.IO.Path]::IsPathRooted($pythonSource)) {
    $pythonSource = Join-Path $repoRoot $pythonSource
}
if (-not (Test-Path $pythonSource)) {
    throw "Python executable not found: $pythonSource"
}

if (-not (Test-Path $envDir)) {
    & $pythonSource -m venv $envDir
    if ($LASTEXITCODE -ne 0) { throw "Failed to create virtual environment." }
}

$python = Join-Path $envDir "Scripts\python.exe"

& $python -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) { throw "Failed to upgrade bootstrap packages." }
& $python -m pip install -r $requirements
if ($LASTEXITCODE -ne 0) { throw "Failed to install Allen prep requirements." }
& $python -m pip install --no-deps brainsets==0.2.0
if ($LASTEXITCODE -ne 0) { throw "Failed to install brainsets without optional dependency tree." }
& $python -m pip install --no-deps -e $repoRoot
if ($LASTEXITCODE -ne 0) { throw "Failed to install predictive_circuit_coding in editable mode." }

Write-Host "Allen visual behavior prep environment ready at $envDir"
Write-Host "Run prep with:"
Write-Host "$python -m predictive_circuit_coding.cli.prepare_data prepare-allen-visual-behavior-neuropixels --config configs/pcc/allen_visual_behavior_neuropixels_local.yaml"
