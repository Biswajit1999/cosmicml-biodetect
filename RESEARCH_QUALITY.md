# Research Quality Upgrade

This repository has been upgraded with a compact research-quality layer: reference anchors, validation checks, and explicit scientific/software boundaries.

## Scope

Physics-informed exoplanet-atmosphere biosignature analysis framework with spectral-feature validation and reproducibility checks.

## Equations And Models

- Transmission depth delta(lambda) approx (Rp(lambda)/Rstar)^2
- Beer-Lambert optical depth tau = integral kappa rho ds

## Reference Anchors

The file `data/research-reference.json` stores benchmark anchors used by `scripts/validate_repository.mjs`. These are intentionally small and auditable so the repository can be checked without network access.

## Reproducibility Upgrade

The validation layer checks source files, reference data, README citations, and incomplete scaffold markers.

## References

- Seager, S., Bains, W. and Petkowski, J.J., 2016. Toward a list of molecules as potential biosignature gases for the search for life on exoplanets and applications to terrestrial biochemistry. Astrobiology, 16(6), pp.465-485.
