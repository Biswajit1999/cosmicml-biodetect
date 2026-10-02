import numpy as np

from cosmicml.benchmark import generate_families, run_benchmark, split_family_ids


def test_generation_is_deterministic():
    first = generate_families(n_families=12, seed=7)
    second = generate_families(n_families=12, seed=7)
    np.testing.assert_allclose(first.spectra, second.spectra)
    np.testing.assert_allclose(first.amplitudes, second.amplitudes)


def test_family_splits_are_disjoint_and_complete():
    ids = np.arange(100)
    splits = split_family_ids(ids)
    concatenated = np.concatenate(list(splits.values()))
    assert len(np.unique(concatenated)) == len(ids)
    assert set(concatenated) == set(ids)
    assert {name: len(values) for name, values in splits.items()} == {
        "train": 50,
        "tune": 20,
        "calibration": 15,
        "test": 15,
    }


def test_benchmark_beats_controls_without_split_overlap():
    result, _ = run_benchmark()
    assert result["split_overlap_count"] == 0
    assert result["macro_test_mae"] < result["macro_mean_baseline_mae"]
    assert result["macro_test_mae"] < result["macro_permuted_label_mae"]


def test_interval_coverage_is_reported_not_assumed():
    result, _ = run_benchmark()
    assert 0 <= result["macro_test_interval_coverage"] <= 1
    assert all(
        0 <= row["test_interval_coverage"] <= 1
        for row in result["species_metrics"]
    )
