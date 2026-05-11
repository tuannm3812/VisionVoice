param(
    [string]$OutputPath = "dist\VisionVoice_cloud_bundle.zip"
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$distDir = Join-Path $repoRoot "dist"
$bundleRoot = Join-Path $distDir "VisionVoice_cloud_bundle"
$zipPath = Join-Path $repoRoot $OutputPath

if (Test-Path $distDir) {
    Remove-Item -LiteralPath $distDir -Recurse -Force
}

New-Item -ItemType Directory -Path $bundleRoot | Out-Null

$itemsToCopy = @(
    "README.md",
    "requirements.txt",
    "notebooks",
    "docs",
    "report",
    "src"
)

foreach ($item in $itemsToCopy) {
    $source = Join-Path $repoRoot $item
    if (Test-Path $source) {
        Copy-Item -LiteralPath $source -Destination $bundleRoot -Recurse -Force
    }
}

$excludedPatterns = @(
    "__pycache__",
    ".ipynb_checkpoints",
    "*.pyc",
    "*.pyo",
    "*.pth",
    "*.pt",
    "*.ckpt",
    "*.log"
)

foreach ($pattern in $excludedPatterns) {
    Get-ChildItem -Path $bundleRoot -Recurse -Force -Filter $pattern -ErrorAction SilentlyContinue |
        Remove-Item -Recurse -Force
}

Compress-Archive -Path (Join-Path $bundleRoot "*") -DestinationPath $zipPath -Force

Write-Host "Created cloud bundle: $zipPath"
