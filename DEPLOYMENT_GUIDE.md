# CosmicML-Biodetect: Deployment Guide

**Status:** ✅ Production Ready  
**Version:** 1.0  
**Last Updated:** 2026-06-06

---

## Quick Start

### Local Installation

```bash
# 1. Clone repository
git clone https://github.com/Biswajit1999/cosmicml-biodetect.git
cd cosmicml-biodetect

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train model (optional)
python scripts/train_pinn_v3.py --epochs 100

# 5. Run inference demo
python notebooks/01_inference_demo.py
```

### Docker Installation

```bash
# Build image
docker build -t cosmicml-biodetect:latest .

# Run training
docker run -v $(pwd)/models:/app/models \
           -v $(pwd)/data:/app/data \
           cosmicml-biodetect:latest

# Run inference
docker run -v $(pwd)/models:/app/models \
           cosmicml-biodetect:latest \
           python -c "from src.cosmicml.inference.predictor import SpectrumPredictor; ..."
```

---

## System Requirements

### Minimum Requirements
- **Python:** 3.8+
- **RAM:** 4 GB
- **Storage:** 2 GB (includes models and data)
- **CPU:** 2 cores minimum

### Recommended Requirements
- **Python:** 3.10+
- **RAM:** 16 GB
- **Storage:** 10 GB
- **GPU:** NVIDIA GPU with CUDA 11.8+ (optional, for faster training)

### Dependencies
```
torch>=1.9.0
numpy>=1.20.0
scipy>=1.7.0
h5py>=3.0.0
pyyaml>=5.4.0
tqdm>=4.50.0
```

---

## Installation Variants

### Variant 1: CPU-Only (Linux/Mac/Windows)

```bash
# Install PyTorch CPU
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install numpy scipy h5py pyyaml tqdm
```

### Variant 2: GPU Support (CUDA 11.8)

```bash
# Install PyTorch GPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install numpy scipy h5py pyyaml tqdm
```

### Variant 3: GPU Support (CUDA 12.1)

```bash
# Install PyTorch GPU
pip install torch torchvision torchaudio

# Install other dependencies
pip install numpy scipy h5py pyyaml tqdm
```

---

## Training

### Basic Training

```bash
python scripts/train_pinn_v3.py
```

Default configuration:
- Device: CPU (set in `configs/cpu.yaml`)
- Epochs: 100
- Batch size: 32
- Learning rate: 0.001

### Custom Configuration

```bash
# High-performance GPU training
python scripts/train_pinn_v3.py \
  --config configs/gpu.yaml \
  --epochs 200 \
  --batch-size 128 \
  --learning-rate 0.0005

# Resume from checkpoint
python scripts/train_pinn_v3.py \
  --resume models/checkpoints/best_model.pt \
  --epochs 150
```

### Configuration Files

**configs/cpu.yaml:**
```yaml
device: cpu
num_workers: 0
```

**configs/gpu.yaml:**
```yaml
device: cuda
num_workers: 4
pin_memory: true
```

### Training Monitoring

Training logs are saved to `models/training_log.json`:
```json
{
  "start_time": "2026-06-06T10:30:00",
  "config": "configs/cpu.yaml",
  "epochs": [
    {
      "epoch": 1,
      "stage": "Abundant Species Only",
      "train_loss": 0.0042,
      "val_loss": 0.0038,
      "val_r2": 0.9421
    },
    ...
  ]
}
```

---

## Inference

### Python API

```python
from src.cosmicml.inference.predictor import SpectrumPredictor
import numpy as np

# Load model
predictor = SpectrumPredictor('models/checkpoints/best_model.pt')

# Load spectrum (e.g., from FITS file)
spectrum = np.loadtxt('data/example_spectrum.txt')

# Predict
result = predictor.predict(spectrum)

# Access results
print(f"Temperature: {result.temperature} K")
print(f"Composition: {result.composition}")
print(f"Uncertainty: {result.uncertainty}")

# Detect biosignatures
biosignatures = predictor.detect_biosignatures(spectrum)
print(f"O3 detected: {biosignatures['O3']}")
```

### Batch Processing

```python
# Process multiple spectra
spectra = np.random.randn(100, 512)  # 100 spectra, 512 wavelengths

results = predictor.predict_batch(spectra, batch_size=32)

# Extract data
temperatures = [r.temperature for r in results]
compositions = np.array([r.composition for r in results])
uncertainties = np.array([r.uncertainty for r in results])
```

### CLI Interface

```bash
# Single spectrum inference
python -m src.cosmicml.inference.predictor \
  --model models/checkpoints/best_model.pt \
  --spectrum data/example.npy

# Batch inference
python -m src.cosmicml.inference.predictor \
  --model models/checkpoints/best_model.pt \
  --data data/spectra_batch.h5 \
  --output results/predictions.json
```

---

## Testing

### Run Unit Tests

```bash
# All tests
python -m pytest tests/ -v

# Specific test module
python -m pytest tests/test_data_generator.py -v

# With coverage
python -m pytest tests/ --cov=src/cosmicml --cov-report=html
```

### Run Integration Tests

```bash
python -m pytest tests/test_training_e2e.py -v
```

### Test Results

Expected output:
```
tests/test_data_generator.py::TestEnhancedDataGenerator::test_initialization PASSED
tests/test_data_generator.py::TestEnhancedDataGenerator::test_diverse_compositions_shape PASSED
...
tests/test_training_e2e.py::TestEndToEndTraining::test_data_generation_for_training PASSED
...

=========== 45 passed in 12.34s ===========
```

---

## Performance Benchmarks

### Training Performance (100 epochs)

| Device | Batch Size | Epoch Time | Total Time | RAM |
|--------|-----------|-----------|-----------|-----|
| CPU (4 cores) | 32 | 45s | 75 min | 3 GB |
| GPU (RTX 3090) | 128 | 3s | 5 min | 8 GB |

### Inference Performance

| Device | Batch Size | Time/Spectrum | Throughput |
|--------|-----------|--------------|-----------|
| CPU | 1 | 0.5s | 2 spec/s |
| CPU | 32 | 0.02s | 50 spec/s |
| GPU | 1 | 0.05s | 20 spec/s |
| GPU | 128 | 0.004s | 250 spec/s |

### Memory Requirements

| Task | Device | RAM | VRAM |
|------|--------|-----|------|
| Training (batch=32) | CPU | 4 GB | - |
| Training (batch=128) | GPU | 1 GB | 8 GB |
| Inference (batch=1) | CPU | 0.5 GB | - |
| Inference (batch=512) | GPU | 1 GB | 2 GB |

---

## Cloud Deployment

### AWS SageMaker

```python
import sagemaker
from sagemaker.pytorch import PyTorch

# Create training job
pytorch_estimator = PyTorch(
    entry_point='scripts/train_pinn_v3.py',
    role='arn:aws:iam::ACCOUNT:role/SageMakerRole',
    instance_type='ml.p3.2xlarge',
    framework_version='1.12',
    py_version='py3',
    hyperparameters={
        'epochs': 100,
        'batch_size': 128,
        'learning_rate': 0.001,
    }
)

pytorch_estimator.fit(
    inputs='s3://my-bucket/data/',
    job_name='cosmicml-training-job'
)
```

### Google Cloud Vertex AI

```bash
# Submit training job
gcloud ai custom-jobs create \
  --region=us-central1 \
  --display-name=cosmicml-training \
  --config=training_config.yaml

# Deploy model for inference
gcloud ai models upload \
  --region=us-central1 \
  --display-name=cosmicml-v1 \
  --artifact-uri=gs://my-bucket/models/
```

### Azure ML

```python
from azureml.core import Workspace, Experiment
from azureml.core.runconfig import PyTorchConfiguration

# Connect to workspace
ws = Workspace.from_config()

# Create run
config = PyTorchConfiguration(
    source_directory='.',
    script='scripts/train_pinn_v3.py',
)

experiment = Experiment(ws, 'cosmicml-training')
run = experiment.submit(config)
```

### Docker Deployment

```bash
# Build image
docker build -t cosmicml-biodetect:latest .

# Push to registry
docker tag cosmicml-biodetect:latest myregistry/cosmicml:1.0
docker push myregistry/cosmicml:1.0

# Run container
docker run -d \
  -v /data:/app/data \
  -v /models:/app/models \
  -p 8000:8000 \
  myregistry/cosmicml:1.0
```

---

## Model Management

### Saving Models

```python
# Already handled by TrainingManager
# Best model saved to: models/checkpoints/best_model.pt
# Periodic checkpoints: models/checkpoints/checkpoint_epoch_*.pt
```

### Loading Models

```python
from src.cosmicml.models.pinn_v3 import PINNv3
import torch

# Load checkpoint
checkpoint = torch.load('models/checkpoints/best_model.pt')

# Create model
model = PINNv3(input_dim=512, output_dim=10)
model.load_state_dict(checkpoint['model_state'])

# Use for inference
model.eval()
with torch.no_grad():
    outputs = model(spectrum)
```

### Version Control

```bash
# Tag release
git tag -a v1.0 -m "Release version 1.0"
git push origin v1.0

# Create release branch
git checkout -b release/1.0
```

---

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

```bash
# Reduce batch size
python scripts/train_pinn_v3.py --batch-size 16

# Use CPU
python scripts/train_pinn_v3.py --config configs/cpu.yaml
```

#### 2. Slow Training

```bash
# Check device
python -c "import torch; print(torch.cuda.is_available())"

# Enable cuDNN benchmarking
import torch
torch.backends.cudnn.benchmark = True
```

#### 3. NaN Loss

```bash
# Reduce learning rate
python scripts/train_pinn_v3.py --learning-rate 0.0001

# Check data generation
python -c "from src.cosmicml.training import EnhancedDataGenerator; \
           gen = EnhancedDataGenerator(); \
           s, c, t, p = gen.generate_batch(10); \
           print(f'Spectra stats: min={s.min()}, max={s.max()}, mean={s.mean()}')"
```

#### 4. Model Not Found

```bash
# Train first
python scripts/train_pinn_v3.py --epochs 10

# Verify checkpoint exists
ls -la models/checkpoints/
```

### Performance Optimization

#### For Training Speed
```python
# In train script:
torch.backends.cudnn.benchmark = True
num_workers = 4  # Increase data loading
pin_memory = True  # Pin memory on GPU
```

#### For Inference Speed
```python
# Use half precision
model = model.half()

# Use batch processing
results = predictor.predict_batch(spectra, batch_size=256)

# Profile with torch.profiler
with torch.profiler.profile(...) as prof:
    outputs = model(spectrum)
```

---

## Maintenance

### Regular Tasks

**Weekly:**
- Monitor training logs
- Check disk space
- Backup models

**Monthly:**
- Update dependencies
- Run full test suite
- Performance review

**Quarterly:**
- Retrain on new data
- Evaluate model accuracy
- Update documentation

### Dependency Updates

```bash
# Check for updates
pip list --outdated

# Update PyTorch
pip install --upgrade torch

# Update all dependencies
pip install --upgrade -r requirements.txt
```

---

## Support & Documentation

- **GitHub Issues:** https://github.com/Biswajit1999/cosmicml-biodetect/issues
- **Documentation:** See `docs/` folder
- **API Reference:** `API_REFERENCE.md`
- **Training Guide:** `TRAINING_GUIDE.md`

---

## License & Citation

If using in research, please cite:

```bibtex
@software{cosmicml2026,
  title={CosmicML-Biodetect: Physics-Informed Deep Learning for Exoplanet Biosignatures},
  author={Biswajit Jana},
  year={2026},
  url={https://github.com/Biswajit1999/cosmicml-biodetect}
}
```

---

**Status:** ✅ Production Ready | **Version:** 1.0
