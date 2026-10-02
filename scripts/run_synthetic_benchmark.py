"""Run and publish the deterministic synthetic inverse benchmark."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cosmicml.benchmark import run_benchmark


def main() -> None:
    result, _ = run_benchmark()
    output = ROOT / "results"
    output.mkdir(exist_ok=True)
    (output / "synthetic_benchmark.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    rows = result["species_metrics"]
    with (output / "species_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    sensitivity = []
    for noise_sigma in (3e-5, 6e-5, 1.2e-4, 2.4e-4):
        trial, _ = run_benchmark(noise_sigma=noise_sigma)
        sensitivity.append(
            {
                "per_replicate_noise_sigma": noise_sigma,
                "macro_test_mae": trial["macro_test_mae"],
                "macro_mean_baseline_mae": trial["macro_mean_baseline_mae"],
                "macro_permuted_label_mae": trial["macro_permuted_label_mae"],
                "macro_test_interval_coverage": trial["macro_test_interval_coverage"],
                "selected_ridge_alpha": trial["selected_ridge_alpha"],
            }
        )
    with (output / "noise_sensitivity.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sensitivity[0].keys())
        writer.writeheader()
        writer.writerows(sensitivity)

    labels = [row["species"] for row in rows]
    x = range(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    axes[0].bar([i - 0.2 for i in x], [row["test_mae"] for row in rows], 0.4, label="Ridge")
    axes[0].bar([i + 0.2 for i in x], [row["mean_baseline_mae"] for row in rows], 0.4, label="Mean baseline")
    axes[0].set_xticks(list(x), labels)
    axes[0].set_ylabel("MAE in injected amplitude")
    axes[0].legend(frameon=False)
    axes[0].set_title("Held-out family recovery")
    axes[1].bar(labels, [row["test_interval_coverage"] for row in rows], color="#217a70")
    axes[1].axhline(0.90, color="#9b3d32", linestyle="--", label="90% target")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Empirical test coverage")
    axes[1].set_title("Split-conformal intervals")
    axes[1].legend(frameon=False)
    axes[2].plot(
        [row["per_replicate_noise_sigma"] for row in sensitivity],
        [row["macro_test_mae"] for row in sensitivity],
        "o-",
        label="Ridge",
    )
    axes[2].axhline(
        result["macro_mean_baseline_mae"],
        color="#777777",
        linestyle=":",
        label="Mean baseline",
    )
    axes[2].set_xscale("log")
    noise_values = [row["per_replicate_noise_sigma"] for row in sensitivity]
    axes[2].set_xticks(noise_values, ["3e-5", "6e-5", "1.2e-4", "2.4e-4"])
    axes[2].xaxis.set_minor_formatter(NullFormatter())
    axes[2].set_xlabel("Per-replicate noise σ")
    axes[2].set_ylabel("Macro test MAE")
    axes[2].set_title("Noise sensitivity")
    axes[2].legend(frameon=False)
    fig.suptitle("Synthetic template-emulator benchmark; not observed-data validation")
    fig.tight_layout()
    fig.savefig(output / "synthetic_benchmark.png", dpi=220)
    plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
