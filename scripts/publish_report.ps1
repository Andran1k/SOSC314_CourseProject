$ErrorActionPreference = "Stop"

$root = Resolve-Path (Join-Path $PSScriptRoot "..")
$docs = Join-Path $root "docs"

if (-not (Test-Path $docs)) {
    New-Item -ItemType Directory -Path $docs | Out-Null
}

Set-Location $root

py -m jupyter nbconvert --to html --execute --no-input notebooks/final_report.ipynb --output index.html --output-dir docs

$noJekyll = Join-Path $docs ".nojekyll"
if (-not (Test-Path $noJekyll)) {
    New-Item -ItemType File -Path $noJekyll | Out-Null
}

Write-Host "Prepared docs/index.html and docs/.nojekyll"
