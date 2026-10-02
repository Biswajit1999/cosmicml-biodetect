# WASP-39 b JWST NIRSpec PRISM data provenance

- Archive: Zenodo record [10.5281/zenodo.6959427](https://doi.org/10.5281/zenodo.6959427)
- Dataset title: *Data & model products from “Identification of carbon dioxide
  in an exoplanet atmosphere”*
- Creator: JWST Transiting Exoplanet Community Early Release Science Team
- Archive object: `JWST_ERS_1st_LOOK_PAPER_DATA.zip`
- Archive MD5: `578368eb0c86014462f109d1e8699693`
- Retrieval represented by this snapshot: 24 September 2026

| Committed file | Original archive entry | LF-normalized SHA-256 |
|---|---|---|
| `wasp39b_eureka_nirspec_prism.ecsv` | `ZENODO/TRANSMISSION_SPECTRA_DATA/EUREKA_REDUCTION.txt` | `18ab790d131d28f4ad97d5a19dc3c5787ecc12269685823bbfd0d38dfe8619a8` |
| `wasp39b_scchimeramodel.txt` | `ZENODO/MODEL_FITS/ScCHIMERA_MODEL.txt` | `45d014400577a5208f9f699a7811c0cc95aeb2c4f308583db9e90297f35cca1d` |
| `wasp39b_scchimeramodel_no_co2.txt` | `ZENODO/MODEL_FITS/ScCHIMERA_MODEL_noCO2.txt` | `7d41ab15efb4350ecce93f7e646d86cf4bd0fdab559f567d8e3dac51374ff004` |

The EUREKA file is a published reduction of the observed transmission spectrum.
The two ScCHIMERA curves were supplied by the publication team as a best-fitting
model and a “remove CO2” sensitivity model. Comparing them is not an independent
retrieval, a nested likelihood-ratio test, or a new molecular detection.

Files are stored as text. Validation normalizes CRLF to LF before hashing so the
archive identity is stable across operating systems.
