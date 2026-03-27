$ErrorActionPreference = "Stop"

Write-Host "=== Installing dependencies ==="
pip install -r requirements.txt
pip install importlib_resources

$CONFIG_V1 = "configs/m2.yaml"
$CONFIG_V2 = "configs/m2_capped.yaml"

Write-Host "=== Ingesting LFW and generating splits ==="
python scripts/ingest_lfw.py --config $CONFIG_V1
if ($LASTEXITCODE -ne 0) { Write-Error "ingest_lfw.py failed"; exit 1 }

Write-Host "=== Generating v1 pairs (baseline) ==="
python scripts/make_pairs.py --config $CONFIG_V1 --data-version v1
if ($LASTEXITCODE -ne 0) { Write-Error "make_pairs.py failed"; exit 1 }

Write-Host "=== Run 1: Baseline threshold sweep on val ==="
python scripts/evaluate.py --config $CONFIG_V1 --split val --sweep --data-version v1 --note "baseline val sweep"
if ($LASTEXITCODE -ne 0) { Write-Error "Run 1 failed"; exit 1 }

Write-Host "=== Run 2: Baseline selected-threshold eval on val ==="
$threshold = (Get-Content "outputs/eval/val/selected_threshold.json" | ConvertFrom-Json).threshold
python scripts/evaluate.py --config $CONFIG_V1 --split val --threshold $threshold --data-version v1 --note "baseline val selected threshold"
if ($LASTEXITCODE -ne 0) { Write-Error "Run 2 failed"; exit 1 }

Write-Host "=== Run 3: Baseline final report on test ==="
python scripts/evaluate.py --config $CONFIG_V1 --split test --threshold $threshold --data-version v1 --note "baseline test final report"
if ($LASTEXITCODE -ne 0) { Write-Error "Run 3 failed"; exit 1 }

Write-Host "=== Generating v2 pairs (capped) ==="
python scripts/make_pairs.py --config $CONFIG_V2 --data-version v2
if ($LASTEXITCODE -ne 0) { Write-Error "make_pairs v2 failed"; exit 1 }

Write-Host "=== Run 4: Post-change threshold sweep on val ==="
python scripts/evaluate.py --config $CONFIG_V2 --split val --sweep --data-version v2 --note "v2 capped val sweep"
if ($LASTEXITCODE -ne 0) { Write-Error "Run 4 failed"; exit 1 }

$threshold_v2 = (Get-Content "outputs/eval_v2/val/selected_threshold.json" | ConvertFrom-Json).threshold

Write-Host "=== Run 5: Post-change test report ==="
python scripts/evaluate.py --config $CONFIG_V2 --split test --threshold $threshold_v2 --data-version v2 --note "v2 capped test final report"
if ($LASTEXITCODE -ne 0) { Write-Error "Run 5 failed"; exit 1 }

Write-Host "=== All 5 runs complete. Results in outputs/runs/ ==="