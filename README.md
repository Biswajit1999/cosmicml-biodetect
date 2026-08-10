# CosmicML-Biodetect: Physics-Informed Neural Networks for Exoplanet Biosignature Detection

[![Tests Status](https://img.shields.io/badge/tests-41%2F41%20passing-brightgreen)](./tests)
[![Code Coverage](https://img.shields.io/badge/coverage-71%25-brightgreen)](./tests)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

## About

CosmicML-Biodetect is a machine learning framework for analyzing exoplanet atmospheres using transmission spectroscopy data. It implements Physics-Informed Neural Networks (PINNs) that enforce atmospheric physics constraints during training, enabling detection of biosignature gases (O₂, CH₄, O₃) from spectroscopic observations. The framework includes a radiative transfer simulator for generating training data, chemistry models for atmospheric composition, and uncertainty quantification via Bayesian inference. Designed for JWST and other space-based spectrometers.

## Overview

CosmicML-Biodetect combines Physics-Informed Neural Networks (PINNs) with Bayesian inference to analyze atmospheric composition from exoplanet transmission spectroscopy data. The framework incorporates atmospheric chemistry and radiative transfer physics directly into neural network training, enabling analysis of spectroscopic observations from JWST, Keck, and HST.

## Approach

The framework uses transmission spectroscopy to infer atmospheric composition. Biosignature gases like oxygen, methane, and ozone are detected by analyzing absorption features in stellar light passing through exoplanet atmospheres. Physics-informed neural networks enforce physical constraints during inference rather than purely data-driven prediction.

## Core Components

- **Physics-Informed Neural Networks (PINNs)**: Combines deep learning with constraint enforcement for physical systems
- **Atmospheric Simulation Engine**: Generates synthetic spectral data using radiative transfer and chemistry models
- **Bayesian Inference Pipeline**: Quantifies uncertainty in composition estimates
- **Real Data Integration**: Compatible with JWST, Keck, and HST observational data formats
- **GPU Support**: Optimized for distributed training on compute clusters

## Installation

### Requirements
- Python 3.9+
- PyTorch 2.0+
- GPU support (NVIDIA CUDA recommended)
- 16GB+ RAM for training

### Quick Start

```bash
git clone https://github.com/Biswajit1999/cosmicml-biodetect.git
cd cosmicml-biodetect
pip install -r requirements.txt
python -m pytest tests/  # Verify installation
```

See [GETTING_STARTED.md](docs/getting_started.md) for detailed setup instructions.

## Project Structure

```
cosmicml-biodetect/
├── src/cosmicml/              # Core library
│   ├── models/                # PINN architectures
│   ├── atmosphere/            # Atmospheric simulation
│   ├── inference/             # Bayesian inference
│   ├── data/                  # Data loading & preprocessing
│   └── utils/                 # Utilities
├── notebooks/                 # Jupyter notebooks for learning
├── scripts/                   # Executable Python scripts
├── data/                      # Data storage
│   ├── raw/                   # Real observational data
│   ├── processed/             # Cleaned data
│   └── simulated/             # Synthetic atmospheres
├── docs/                      # Documentation
├── tests/                     # Unit & integration tests
├── configs/                   # Configuration files
└── models/                    # Trained model checkpoints
```

## Quick Start: Running the Pipeline

### 1. Generate Synthetic Training Data
```bash
python scripts/generate_synthetic_data.py \
  --num_atmospheres 10000 \
  --output_dir data/simulated/
```

### 2. Train the PINN Model
```bash
python scripts/train_pinn.py \
  --config configs/gpu.yaml \
  --data_dir data/simulated/ \
  --output_dir models/
```

### 3. Analyze Real Exoplanet Spectra
```bash
python scripts/predict_biosignatures.py \
  --model models/pinn_trained.pt \
  --spectra data/raw/jwst_observations.fits \
  --output results/detections.json
```

See notebooks for interactive tutorials.

## Documentation

- **[Theory & Methodology](docs/theory.md)** - Deep dive into PINNs, atmospheric chemistry, and Bayesian inference
- **[Getting Started](docs/getting_started.md)** - Installation, environment setup, and first experiments
- **[API Reference](docs/api_reference.md)** - Complete module documentation
- **[Research Brief](../CosmicML_Biodetect_Research_Brief.docx)** - Comprehensive introduction for beginners

## Jupyter Notebooks

Interactive learning guides:

1. **[01_Introduction.ipynb](notebooks/01_introduction.ipynb)** - Overview of exoplanet atmospheres and biosignatures
2. **[02_Data_Exploration.ipynb](notebooks/02_data_exploration.ipynb)** - Visualizing spectral data
3. **[03_PINN_Training.ipynb](notebooks/03_pinn_training.ipynb)** - Building and training a PINN model
4. **[04_Biosignature_Detection.ipynb](notebooks/04_biosignature_detection.ipynb)** - Bayesian inference for life detection
5. **[05_JWST_Analysis.ipynb](notebooks/05_jwst_analysis.ipynb)** - Real observations from James Webb

## Core Modules

### `cosmicml.atmosphere`
Simulates exoplanet atmospheres under various conditions:
- **Radiative transfer calculations** for spectral generation
- **Chemical kinetics** for reaction networks
- **Multiple planetary scenarios** (habitable zones, different star types, etc.)

### `cosmicml.models`
Physics-informed neural network implementations:
- **PINN architecture** with physics loss terms
- **Encoder/Decoder networks** for dimension reduction
- **Constraint layers** enforcing chemical equations

### `cosmicml.inference`
Bayesian analysis tools:
- **Likelihood functions** for spectroscopic data
- **MCMC sampling** for posterior estimation
- **Uncertainty quantification** for biosignature probabilities

### `cosmicml.data`
Data handling and preprocessing:
- **JWST data pipeline** for real observations
- **Synthetic data generation** with configurable parameters
- **Normalization and feature engineering**

## Computational Requirements

- **Training on 10K atmospheres:** ~8-12 hours on NVIDIA A100 GPU
- **Inference on single spectrum:** ~0.1 seconds
- **Memory requirements:** 16GB RAM minimum; 32GB+ recommended for large batches

## Key Publications & References

This project builds on:
- Raissi et al. (2019) - Physics-Informed Neural Networks
- Kawashima et al. (2021) - Exoplanet atmosphere characterization
- Madhusudhan et al. (2022) - Biosignature detection in transmission spectra

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Areas where help is needed:
- Additional atmospheric chemistry modules
- PINN architecture improvements
- Real data pipelines for Keck and HST
- Performance optimization for GPU clusters
- Educational materials and tutorials

## Citation

If you use CosmicML-Biodetect in your research, please cite:

```bibtex
@software{cosmicml2024,
  title={CosmicML-Biodetect: Physics-Informed Neural Networks for Exoplanet Biosignature Detection},
  author={Your Name and Contributors},
  year={2024},
  url={https://github.com/Biswajit1999/cosmicml-biodetect}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details

## Contact & Support

- **Documentation Issues:** Check [docs/](docs/)
- **Bug Reports:** Create an [issue](https://github.com/Biswajit1999/cosmicml-biodetect/issues)
- **Questions:** Start a [discussion](https://github.com/Biswajit1999/cosmicml-biodetect/discussions)

---

**Last Updated:** 2024  
**Status:** Active Development  
**Collaboration:** Open to partnerships with astronomy groups and ML researchers

## Research Quality Upgrade

See [RESEARCH_QUALITY.md](RESEARCH_QUALITY.md) for the validation layer, reference anchors, equations and research boundaries added to this repository.
