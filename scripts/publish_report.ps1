$ErrorActionPreference = "Stop"

$root = Resolve-Path (Join-Path $PSScriptRoot "..")
$docs = Join-Path $root "docs"
$siteNotebookDir = Join-Path $docs "notebooks"
$siteFiguresDir = Join-Path $docs "figures"
$sourceNotebook = Join-Path $root "notebooks\final_report.ipynb"
$renderedIndex = Join-Path $docs "index.html"

if (-not (Test-Path $docs)) {
    New-Item -ItemType Directory -Path $docs | Out-Null
}

foreach ($dir in @($siteNotebookDir, $siteFiguresDir)) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir | Out-Null
    }
}

Set-Location $root

py -m jupyter nbconvert --to html --execute --no-input notebooks/final_report.ipynb --output index.html --output-dir docs

Copy-Item -Path $sourceNotebook -Destination (Join-Path $siteNotebookDir "final_report.ipynb") -Force
Copy-Item -Path (Join-Path $root "figures\*") -Destination $siteFiguresDir -Recurse -Force

$html = Get-Content $renderedIndex -Raw
$html = $html.Replace("../notebooks/", "notebooks/")
$html = $html.Replace("../figures/", "figures/")
[System.IO.File]::WriteAllText(
    $renderedIndex,
    $html,
    [System.Text.UTF8Encoding]::new($false)
)

$noJekyll = Join-Path $docs ".nojekyll"
if (-not (Test-Path $noJekyll)) {
    New-Item -ItemType File -Path $noJekyll | Out-Null
}

Write-Host "Prepared docs/index.html, docs/.nojekyll, docs/notebooks/, and docs/figures/"
