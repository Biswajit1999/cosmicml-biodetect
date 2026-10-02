#!/usr/bin/env python
"""Audit a published WASP-39 b JWST spectrum against supplied model curves."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "real"
RESULTS = ROOT / "results"


def fitted_offset(observed: np.ndarray, model: np.ndarray, sigma: np.ndarray) -> float:
    weights = sigma**-2
    return float(np.sum(weights * (observed - model)) / np.sum(weights))


def compare_model(
    observed: np.ndarray, model: np.ndarray, sigma: np.ndarray
) -> tuple[float, float, np.ndarray]:
    offset = fitted_offset(observed, model, sigma)
    residual = (observed - (model + offset)) / sigma
    return offset, float(np.sum(residual**2)), residual


def main() -> None:
    observed = np.genfromtxt(
        DATA / "wasp39b_eureka_nirspec_prism.ecsv",
        names=True,
        comments="#",
        skip_header=12,
    )
    full_curve = np.loadtxt(DATA / "wasp39b_scchimeramodel.txt", comments="#")
    no_co2_curve = np.loadtxt(
        DATA / "wasp39b_scchimeramodel_no_co2.txt", comments="#"
    )

    wavelength = observed["wavelength"]
    depth = observed["tr_depth"]
    sigma = 0.5 * (observed["tr_depth_errneg"] + observed["tr_depth_errpos"])
    overlap = (
        (wavelength >= max(full_curve[:, 0].min(), no_co2_curve[:, 0].min()))
        & (wavelength <= min(full_curve[:, 0].max(), no_co2_curve[:, 0].max()))
        & np.isfinite(depth)
        & np.isfinite(sigma)
        & (sigma > 0)
    )
    wavelength, depth, sigma = wavelength[overlap], depth[overlap], sigma[overlap]
    full = np.interp(wavelength, full_curve[:, 0], full_curve[:, 1])
    no_co2 = np.interp(wavelength, no_co2_curve[:, 0], no_co2_curve[:, 1])

    flat = np.repeat(np.average(depth, weights=sigma**-2), len(depth))
    _, chi2_flat, _residual_flat = compare_model(depth, flat, sigma)
    offset_full, chi2_full, residual_full = compare_model(depth, full, sigma)
    offset_no, chi2_no, residual_no = compare_model(depth, no_co2, sigma)
    delta_chi2 = chi2_no - chi2_full

    rows = []
    for block_index, block in enumerate(np.array_split(np.arange(len(depth)), 8)):
        keep = np.ones(len(depth), dtype=bool)
        keep[block] = False
        _, full_deleted, _ = compare_model(depth[keep], full[keep], sigma[keep])
        _, no_deleted, _ = compare_model(depth[keep], no_co2[keep], sigma[keep])
        rows.append(
            {
                "omitted_block": block_index + 1,
                "wavelength_min_um": float(wavelength[block].min()),
                "wavelength_max_um": float(wavelength[block].max()),
                "n_retained": int(keep.sum()),
                "delta_chi2_no_co2_minus_full": float(no_deleted - full_deleted),
            }
        )

    co2_band = (wavelength >= 4.0) & (wavelength <= 4.6)
    point_delta = residual_no**2 - residual_full**2
    summary = {
        "dataset": "WASP-39 b EUREKA JWST NIRSpec PRISM transmission spectrum",
        "archive_doi": "10.5281/zenodo.6959427",
        "comparison_scope": "supplied model sensitivity audit; not an independent retrieval",
        "n_observed_bins": len(observed),
        "n_common_model_bins": len(depth),
        "wavelength_min_um": float(wavelength.min()),
        "wavelength_max_um": float(wavelength.max()),
        "fitted_offset_full_depth_fraction": offset_full,
        "fitted_offset_no_co2_depth_fraction": offset_no,
        "chi2_flat": chi2_flat,
        "chi2_full_supplied_model": chi2_full,
        "chi2_no_co2_supplied_model": chi2_no,
        "delta_chi2_no_co2_minus_full": delta_chi2,
        "delta_chi2_from_4p0_to_4p6_um": float(point_delta[co2_band].sum()),
        "block_deletion_delta_chi2_min": float(
            min(row["delta_chi2_no_co2_minus_full"] for row in rows)
        ),
        "block_deletion_delta_chi2_max": float(
            max(row["delta_chi2_no_co2_minus_full"] for row in rows)
        ),
        "interpretation": (
            "Within the publication-supplied model pair, the full curve is preferred. "
            "This re-analysis does not reproduce the atmospheric retrieval or detection significance."
        ),
    }

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "wasp39b_real_spectrum_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (RESULTS / "wasp39b_block_deletions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    fig, (ax, residual_ax) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]}
    )
    ax.errorbar(
        wavelength,
        depth * 100,
        yerr=sigma * 100,
        fmt="o",
        ms=3.5,
        color="#172a3a",
        ecolor="#71808d",
        alpha=0.82,
        label="Published EUREKA reduction",
    )
    ax.plot(wavelength, (full + offset_full) * 100, color="#147d73", lw=2, label="Supplied full model + offset")
    ax.plot(wavelength, (no_co2 + offset_no) * 100, color="#b44d3a", lw=1.8, label="Supplied remove-CO₂ model + offset")
    ax.axvspan(4.0, 4.6, color="#d7b65d", alpha=0.16, label="4.0–4.6 μm audit band")
    ax.set_ylabel("Transit depth [%]")
    ax.set_title("WASP-39 b · published JWST/NIRSpec PRISM spectrum")
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    residual_ax.axhline(0, color="#4b5560", lw=1)
    residual_ax.plot(wavelength, residual_full, "o-", ms=3, lw=1, color="#147d73", label="Full")
    residual_ax.plot(wavelength, residual_no, "o-", ms=3, lw=1, color="#b44d3a", alpha=0.78, label="Remove CO₂")
    residual_ax.set_xlabel("Wavelength [μm]")
    residual_ax.set_ylabel("Residual / σ")
    residual_ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(RESULTS / "wasp39b_real_spectrum.png", dpi=220)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
