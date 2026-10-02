"""Validate public claims and reproduced benchmark evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cosmicml.benchmark import run_benchmark


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


readme = (ROOT / "README.md").read_text(encoding="utf-8")
for forbidden in (
    "95%+ accuracy",
    "41/41 passing",
    "Real Data Integration",
    "Ready for deployment",
    "Research Quality Upgrade",
):
    require(forbidden not in readme, f"unsupported public claim remains: {forbidden}")

for removed in (
    "MAJOR_UPGRADE_STATUS.md",
    "RESEARCH_UPGRADE_PLAN.md",
    "UPGRADE_PROGRESS.md",
    "paper/RESEARCH_PAPER.md",
):
    require(not (ROOT / removed).exists(), f"superseded claim artifact remains: {removed}")

require(not list((ROOT / "models").glob("*.ckpt")), "untraceable checkpoints remain")
for path, token in (
    ("src/cosmicml/data/loader.py", "Observed JWST ingestion is not implemented"),
    ("src/cosmicml/data/jwst_pipeline.py", "No synthetic spectrum is substituted"),
    ("src/cosmicml/inference/bayesian.py", "zero-filled pseudo-posteriors"),
):
    require(token in (ROOT / path).read_text(encoding="utf-8"), f"missing fail-closed guard: {path}")

stored = json.loads((ROOT / "results/synthetic_benchmark.json").read_text(encoding="utf-8"))
fresh, _ = run_benchmark()
require(stored["split_overlap_count"] == 0, "stored split overlap must be zero")
require(stored["split_counts"] == fresh["split_counts"], "stored split counts drifted")
for key in (
    "macro_test_mae",
    "macro_mean_baseline_mae",
    "macro_permuted_label_mae",
    "macro_test_interval_coverage",
):
    require(np.isclose(stored[key], fresh[key], rtol=1e-9, atol=1e-12), f"metric drift: {key}")
require(stored["macro_test_mae"] < stored["macro_mean_baseline_mae"], "model does not beat mean baseline")
require(stored["macro_test_mae"] < stored["macro_permuted_label_mae"], "model does not beat permutation control")

with (ROOT / "results/noise_sensitivity.csv").open(encoding="utf-8") as handle:
    sensitivity = list(csv.DictReader(handle))
require(len(sensitivity) == 4, "noise sensitivity must contain four levels")
noise = np.array([float(row["per_replicate_noise_sigma"]) for row in sensitivity])
mae = np.array([float(row["macro_test_mae"]) for row in sensitivity])
require(np.all(np.diff(noise) > 0), "noise levels must increase")
require(np.all(np.diff(mae) > 0), "reported MAE must worsen across the declared noise sweep")

figure = (ROOT / "results/synthetic_benchmark.png").read_bytes()
require(figure.startswith(b"\x89PNG\r\n\x1a\n"), "benchmark figure is not a PNG")

source_hashes = {
    "wasp39b_eureka_nirspec_prism.ecsv": "18ab790d131d28f4ad97d5a19dc3c5787ecc12269685823bbfd0d38dfe8619a8",
    "wasp39b_scchimeramodel.txt": "45d014400577a5208f9f699a7811c0cc95aeb2c4f308583db9e90297f35cca1d",
    "wasp39b_scchimeramodel_no_co2.txt": "7d41ab15efb4350ecce93f7e646d86cf4bd0fdab559f567d8e3dac51374ff004",
}
for filename, expected in source_hashes.items():
    raw = (ROOT / "data/real" / filename).read_bytes()
    canonical = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    require(hashlib.sha256(canonical).hexdigest() == expected, f"source digest drift: {filename}")

real = json.loads((ROOT / "results/wasp39b_real_spectrum_summary.json").read_text(encoding="utf-8"))
require(real["n_observed_bins"] == 94, "unexpected observed-bin count")
require(real["n_common_model_bins"] == 93, "unexpected common model-bin count")
require(real["delta_chi2_no_co2_minus_full"] > 0, "supplied-model preference changed sign")
require(
    0 < real["block_deletion_delta_chi2_min"] <= real["block_deletion_delta_chi2_max"],
    "invalid block-deletion sensitivity range",
)
real_figure = (ROOT / "results/wasp39b_real_spectrum.png").read_bytes()
require(real_figure.startswith(b"\x89PNG\r\n\x1a\n"), "real-spectrum figure is not a PNG")
print(
    "Evidence valid: 360 disjoint families; "
    f"test MAE {stored['macro_test_mae']:.5f}; "
    f"coverage {stored['macro_test_interval_coverage']:.4f}; "
    "94 published JWST bins verified."
)
