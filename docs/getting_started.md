# Getting Started with CosmicML-Biodetect

This guide will help you set up the project and run your first biosignature detection pipeline.

## Prerequisites

- Python 3.9 or higher
- 16GB RAM minimum (32GB recommended)
- NVIDIA GPU with CUDA support (optional but highly recommended)
- Git

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/cosmicml-biodetect.git
cd cosmicml-biodetect
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**For GPU support (CUDA 11.8):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 4: Install the Package

```bash
pip install -e .
```

### Step 5: Verify Installation

```bash
python -c "import cosmicml; print('Installation successful!')"
pytest tests/ -v
```

## Your First Spectrum Analysis

### Using Jupyter Notebook (Recommended for Learning)

```bash
jupyter lab
```

Then open `notebooks/01_introduction.ipynb` to learn the basics.

### Command-Line Quick Start

#### 1. Generate Synthetic Training Data

```bash
python scripts/generate_synthetic_data.py \
  --num_atmospheres 1000 \
  --output_dir data/simulated/ \
  --num_workers 4
```

This creates 1000 synthetic exoplanet atmospheres with associated spectra.
- Takes ~2-5 minutes on GPU
- Takes ~30 minutes on CPU

#### 2. Train a PINN Model

```bash
python scripts/train_pinn.py \
  --config configs/gpu.yaml \
  --data_dir data/simulated/ \
  --batch_size 32 \
  --epochs 50 \
  --output_dir models/
```

This trains a Physics-Informed Neural Network on your synthetic data.
- Takes ~1-2 hours on NVIDIA A100
- Takes ~12-24 hours on NVIDIA V100
- Takes ~2-3 days on CPU (not recommended)

#### 3. Detect Biosignatures

```bash
python scripts/predict_biosignatures.py \
  --model models/pinn_trained.pt \
  --spectra data/sample_jwst.fits \
  --output results/detections.json
```

This analyzes real (or simulated) exoplanet spectra and predicts biosignatures.

## Understanding the Output

### Generated Files

**After data generation:**
- `data/simulated/atmospheres.h5` - Atmospheric compositions
- `data/simulated/spectra.h5` - Transmission spectra
- `data/simulated/metadata.json` - Planetary parameters

**After training:**
- `models/pinn_trained.pt` - Trained neural network weights
- `models/training_log.csv` - Loss curves and metrics
- `results/figures/` - Training curves and analysis plots

**After biosignature detection:**
- `results/detections.json` - Predictions with probabilities
- `results/figures/` - Spectrum analysis visualizations

## Configuration Files

The project includes pre-configured settings for different environments:

### `configs/gpu.yaml` - For GPU Training
```yaml
device: cuda
batch_size: 64
num_workers: 4
learning_rate: 0.001
epochs: 100
```

### `configs/cpu.yaml` - For CPU-Only Systems
```yaml
device: cpu
batch_size: 8
num_workers: 1
learning_rate: 0.001
epochs: 50  # Fewer epochs due to slower training
```

## Jupyter Notebooks

Interactive tutorials are provided:

1. **01_introduction.ipynb** - Learn about exoplanet atmospheres and biosignatures
2. **02_data_exploration.ipynb** - Visualize synthetic spectra
3. **03_pinn_training.ipynb** - Train a PINN model with detailed explanations
4. **04_biosignature_detection.ipynb** - Run Bayesian inference
5. **05_jwst_analysis.ipynb** - Analyze real JWST data

Start with `01_introduction.ipynb` if you're new to the subject.

## Next Steps

### For Scientists
- Read `docs/theory.md` for the physics behind PINNs
- Explore the sample data in `data/simulated/`
- Try modifying `scripts/generate_synthetic_data.py` to test new atmospheric conditions
- Compare results with published exoplanet studies

### For Machine Learning Engineers
- Study the PINN implementation in `src/cosmicml/models/pinn.py`
- Experiment with different architectures in `src/cosmicml/models/`
- Profile the code to find optimization opportunities
- Run benchmarks on different hardware

### For Educators
- Use the notebooks in a classroom setting
- Create derived tutorials for specific topics
- Contribute educational materials (see CONTRIBUTING.md)

## Troubleshooting

### Import Errors
```
ModuleNotFoundError: No module named 'cosmicml'
```
Solution: Make sure you've run `pip install -e .` in the project directory.

### CUDA Errors
```
RuntimeError: CUDA out of memory
```
Solution: Reduce `batch_size` in configs or use `device: cpu` in configs/cpu.yaml

### Data Not Found
```
FileNotFoundError: data/simulated/spectra.h5
```
Solution: Run `python scripts/generate_synthetic_data.py` first.

### Slow Training
If training is very slow:
- Check you're using GPU: `python -c "import torch; print(torch.cuda.is_available())"`
- Reduce epochs in config file
- Use a smaller dataset with `--num_atmospheres 500`

## System Requirements by Task

| Task | Minimum | Recommended | Ideal |
|------|---------|-------------|-------|
| Data generation | 8GB RAM, 1 core | 16GB RAM, 4 cores | 32GB RAM, 8 cores + GPU |
| PINN training | 12GB VRAM | 24GB VRAM | 40GB+ VRAM (A100/H100) |
| Inference | 4GB VRAM | 8GB VRAM | 16GB+ VRAM |
| Jupyter notebooks | 8GB RAM | 16GB RAM | 32GB RAM |

## GPU Benchmarks

Training time for 10,000 synthetic atmospheres on different hardware:

| Hardware | Training Time | Cost |
|----------|---------------|------|
| CPU (Intel i7) | ~2-3 days | $0 (if own) |
| GPU (NVIDIA V100) | ~8 hours | $2-4/hour (cloud) |
| GPU (NVIDIA A100) | ~2 hours | $3-5/hour (cloud) |
| GPU (NVIDIA H100) | ~1 hour | $4-7/hour (cloud) |

## Cloud Computing

### Google Colab (Free)
```python
# Install in notebook
!pip install -r requirements.txt
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### AWS EC2
Launch a GPU instance (p3.2xlarge with V100) and follow the installation steps.

### Azure
Use Azure Machine Learning with GPU compute targets.

## Getting Help

- **Documentation**: See `docs/` folder
- **Issues**: Post on GitHub Issues with reproducible example
- **Discussions**: Start a Discussion for general questions
- **Notebooks**: Interactive tutorials in `notebooks/`

---

Happy analyzing! May you discover signs of life in the cosmos. 🌟
