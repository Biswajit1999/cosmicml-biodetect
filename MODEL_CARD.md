# Model card: analytic template-amplitude ridge benchmark

## Intended use

The model is a methodological control for a synthetic inverse problem. It can
test data partitioning, baseline comparison, negative controls, interval
calibration, and artifact regeneration in a small CPU-only workflow.

## Out-of-scope use

Do not use this model to classify observed spectra, claim a molecular
detection, rank inhabited planets, estimate a probability of life, guide
telescope time, or make statements about JWST/HST/Keck sensitivity.

## Inputs and targets

- Input: 256-point family-averaged synthetic spectrum over 0.5–5.0 μm.
- Target: five dimensionless injected analytic-template amplitudes in [0, 1].
- Species labels: H2O, CO2, O2, O3, CH4.
- Replication: four independent Gaussian-noise realizations averaged per latent
  family.

The labels name template families for interpretability. They are not retrieved
physical abundances because the generator is not a validated radiative-
transfer calculation.

## Evaluation protocol

The 360 latent families are assigned to mutually exclusive train (180), tune
(72), calibration (54), and test (54) partitions with fixed seeds. Ridge alpha
is chosen on tune data. Conformal residuals come only from calibration data.
Reported accuracy and interval coverage come only from the final test data.

Controls are a train+tune target mean and a ridge model fitted after permuting
the training targets. A sensitivity sweep repeats the complete procedure at
four per-replicate noise scales.

## Primary result

At noise sigma `6e-5`, macro test MAE is 0.01375, compared with 0.20846 for
the mean baseline and 0.22352 for the permuted-label control. Nominal 90%
split-conformal intervals cover 0.9407 of the held-out target entries.

The good score is expected because train, calibration, and test data share the
same analytic generator and band locations. It must not be extrapolated to
line-by-line simulations or observations.

The separate WASP-39 b script compares a published spectrum with two
publication-supplied model curves. It does not apply this ridge model to those
data and is not evidence of observational generalization.

## Limitations

- no clouds, hazes, stellar heterogeneity, correlated detector noise, or
  wavelength-calibration uncertainty;
- no line lists, pressure broadening, chemistry, or physically coupled
  abundances;
- random in-distribution family holdout, not cross-instrument or cross-target
  validation;
- intervals address exchangeability within this emulator, not forward-model
  misspecification;
- only one seed is designated as the primary analysis.

## Reproducibility

The generator, split seeds, alpha grid, calibration rule, metrics, and output
formats are code-defined in `src/cosmicml/benchmark.py`. CI regenerates the
artifacts and fails if tracked outputs change.
