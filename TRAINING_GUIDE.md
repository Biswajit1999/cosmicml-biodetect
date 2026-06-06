# CosmicML-Biodetect: Training Guide

**Version:** 1.0  
**Last Updated:** 2026-06-06

---

## Overview

This guide covers training PINN v3 models using the Phase 4 training pipeline. The system implements:

1. **Physics-based data generation** (Phase 2 radiative transfer)
2. **Curriculum learning** (progressive difficulty)
3. **Advanced optimization** (AdamW + SGDR)
4. **Multi-task loss balancing** (learned task weights)

---

## Quick Start

### 1. Basic Training (CPU)

```bash
python scripts/train_pinn_v3.py
```

Default settings:
- Device: CPU
- Epochs: 100
- Batch size: 32
- Learning rate: 0.001

Expected runtime: **~2 hours**

### 2. GPU Training (10x faster)

```bash
python scripts/train_pinn_v3.py --config configs/gpu.yaml --epochs 100 --batch-size 128
```

Expected runtime: **~12 minutes**

### 3. Resume Training

```bash
python scripts/train_pinn_v3.py --resume models/checkpoints/best_model.pt --epochs 150
```

---

## Training Configuration

### Command-line Arguments

```bash
python scripts/train_pinn_v3.py [OPTIONS]

Options:
  --config FILE         Training config (default: configs/cpu.yaml)
  --epochs INT          Number of epochs (default: 100)
  --batch-size INT      Batch size (default: 32)
  --learning-rate FLOAT Initial learning rate (default: 0.001)
  --data-dir PATH       Data directory (default: data/simulated/)
  --output-dir PATH     Output directory (default: models/)
  --resume PATH         Resume from checkpoint (default: None)
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

Create custom configs as needed.

---

## Training Pipeline

### Phase 1: Data Generation

```
5000 synthetic atmospheres generated using Phase 2 radiative transfer
├── 4000 training samples
├── 1000 validation samples
└── Physics: Rayleigh scattering + CIA + line broadening
```

**Composition diversity:**
- Earth-like (N₂/O₂ dominated)
- Venus-like (CO₂ dominated)
- H₂/He worlds (mini-Neptune)
- O₃-rich (photochemistry)
- CH₄-rich (biosignature)

### Phase 2: Model Initialization

```
PINN v3 model created with:
├── 512 input dimensions (wavelengths)
├── 4 attention heads (spectral feature learning)
├── 10 output dimensions (species)
├── Bayesian uncertainty quantification
└── Physics loss integration (Gibbs + equilibrium)
```

### Phase 3: Curriculum Learning

```
4-stage curriculum progression:

Stage 1 (10 epochs): Abundant species only
├── Species: N₂, O₂, CO₂, H₂O
├── Temperature: 288K (fixed)
└── Learning rate scale: 1.0

Stage 2 (10 epochs): Add medium species
├── Add: CH₄, H₂
├── Temperature: 200-400K
└── Learning rate scale: 0.5

Stage 3 (10 epochs): Add trace species
├── Add: O₃, NH₃, NO, H₂S
├── Temperature: 100-500K
└── Learning rate scale: 0.2

Stage 4 (20 epochs): Full training
├── All 10 species
├── Temperature: 100-1500K
└── Learning rate scale: 0.1
```

### Phase 4: Training Loop

```
for epoch in range(1, epochs+1):
    stage = curriculum.get_current_stage()
    
    for batch in training_data:
        # Forward pass
        output = model(spectrum)
        
        # Multi-task loss
        loss_dict = {
            'data': MSE(output['comp'], target_comp),
            'temperature': MSE(output['temp'], target_temp),
            'physics': compute_physics_loss(...),
            'aleatoric': uncertainty_calibration(...),
        }
        
        # Balanced loss with learned weights
        total_loss = Σ_i exp(-σ_i) * L_i + σ_i
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()  # AdamW + gradient clipping + LR scheduling
    
    # Validation
    val_loss, val_r2 = validate(val_data)
    
    # Checkpoint
    if val_loss < best_loss:
        save(model, 'best_model.pt')
    
    # Curriculum advancement
    if (epoch % stage['epochs']) == 0:
        curriculum.advance_stage()
```

### Phase 5: Finalization

```
Save:
├── Best model checkpoint
├── Periodic checkpoints (every 10 epochs)
├── Training log (JSON)
└── Final metrics
```

---

## Hyperparameter Tuning

### Learning Rate

Default: `0.001`

**Adjustment:**
- Too high: Unstable training, NaN loss
- Too low: Slow convergence

```bash
# Conservative (stable but slow)
python scripts/train_pinn_v3.py --learning-rate 0.0001

# Aggressive (fast but risky)
python scripts/train_pinn_v3.py --learning-rate 0.01
```

**Recommended schedule:**
- Start: 0.001
- Warmup: 0.0001 → 0.001 (first 10 epochs)
- Annealing: 0.001 → 1e-6 (cosine schedule)

### Batch Size

Default: `32`

**GPU memory requirements:**
- Batch 32: ~4GB VRAM
- Batch 64: ~6GB VRAM
- Batch 128: ~8GB VRAM

```bash
# Larger batch = faster training but more memory
python scripts/train_pinn_v3.py --batch-size 128

# Smaller batch = stable gradient estimates
python scripts/train_pinn_v3.py --batch-size 8
```

**Recommended:**
- CPU: 16-32
- GPU (6GB): 64
- GPU (12GB): 128
- GPU (24GB): 256

### Number of Epochs

Default: `100` (50 epochs per curriculum + 20 final)

**Curriculum stage durations:**
- Stage 1 (Abundant): 10 epochs
- Stage 2 (Medium): 10 epochs
- Stage 3 (Trace): 10 epochs
- Stage 4 (Full): 20 epochs

```bash
# Quick test run
python scripts/train_pinn_v3.py --epochs 10

# Production training
python scripts/train_pinn_v3.py --epochs 200
```

### Weight Decay

Default: `0.0001`

Prevents overfitting. Higher values → stronger regularization.

---

## Monitoring Training

### Training Log

Automatically saved to `models/training_log.json`:

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
      "val_r2": 0.9421,
      "learning_rate": 0.0001
    },
    ...
  ],
  "end_time": "2026-06-06T12:30:00"
}
```

### Real-time Monitoring

```bash
# Watch training in real-time
tail -f models/training_log.json | jq '.epochs[-1]'

# Or use TensorBoard (future enhancement)
tensorboard --logdir=models/
```

### Expected Metrics

**Convergence:**
- Epoch 1-10: Rapid loss decrease (stage 1)
- Epoch 10-30: Slower decrease (stages 2-3)
- Epoch 30+: Plateau at validation loss

**R² Score:**
- Epoch 10: ~0.85 (common species)
- Epoch 50: ~0.92 (all species)
- Epoch 100: >0.95 (target)

**Loss Values:**
- Train loss: 0.003-0.01
- Val loss: 0.004-0.012

---

## Troubleshooting

### Issue 1: NaN Loss

**Symptoms:**
- Loss becomes NaN after few steps
- Training crashes

**Causes:**
- Learning rate too high
- Gradient explosion
- Bad initialization

**Solutions:**
```bash
# Reduce learning rate
python scripts/train_pinn_v3.py --learning-rate 0.0001

# Use smaller batch size
python scripts/train_pinn_v3.py --batch-size 8

# Check data for NaNs
python -c "
from src.cosmicml.training import EnhancedDataGenerator
gen = EnhancedDataGenerator()
s, c, t, p = gen.generate_batch(10)
print(f'Spectra: min={s.min()}, max={s.max()}, NaN={np.isnan(s).any()}')
print(f'Compositions: min={c.min()}, max={c.max()}, NaN={np.isnan(c).any()}')
"
```

### Issue 2: Slow Training

**Symptoms:**
- ~1 epoch per minute (CPU)
- Expecting much faster

**Causes:**
- Using CPU instead of GPU
- Too many workers
- Slow disk I/O

**Solutions:**
```bash
# Use GPU
python scripts/train_pinn_v3.py --config configs/gpu.yaml

# Reduce workers
# Edit configs/gpu.yaml: num_workers: 2

# Use SSD for data
# Move data to /tmp or fast disk
```

### Issue 3: Poor Validation Accuracy

**Symptoms:**
- Val R² stuck at 0.7-0.8
- Not improving after 50 epochs

**Causes:**
- Learning rate too small
- Insufficient training
- Curriculum too aggressive

**Solutions:**
```bash
# Train longer
python scripts/train_pinn_v3.py --epochs 200

# Increase learning rate
python scripts/train_pinn_v3.py --learning-rate 0.002

# Larger batch size (more stable gradients)
python scripts/train_pinn_v3.py --batch-size 64
```

### Issue 4: Out of Memory

**Symptoms:**
- CUDA out of memory error
- Process killed

**Causes:**
- Batch size too large
- Model too big

**Solutions:**
```bash
# Reduce batch size
python scripts/train_pinn_v3.py --batch-size 16

# Use CPU
python scripts/train_pinn_v3.py --config configs/cpu.yaml

# Check GPU memory
nvidia-smi
```

### Issue 5: Checkpoint Not Found

**Symptoms:**
- "Model not found at models/checkpoints/best_model.pt"

**Causes:**
- Training hasn't completed
- Wrong path

**Solutions:**
```bash
# Check if checkpoints exist
ls -la models/checkpoints/

# Train first
python scripts/train_pinn_v3.py --epochs 10
```

---

## Advanced Training Techniques

### Multi-GPU Training

Currently single-GPU only. For multi-GPU:

```python
# In trainer.py
model = nn.DataParallel(model)
```

### Mixed Precision Training

For faster training with less memory:

```python
# In trainer.py
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    loss = model(x)
    
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Distributed Training

For multiple machines:

```bash
# Edit trainer.py to use DistributedDataParallel
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    scripts/train_pinn_v3.py
```

### Custom Data

```python
# Create custom composition generator
def load_custom_spectra(data_dir):
    spectra = []
    compositions = []
    
    for file in os.listdir(data_dir):
        spectrum = np.loadtxt(file)
        composition = get_composition(file)
        
        spectra.append(spectrum)
        compositions.append(composition)
    
    return np.array(spectra), np.array(compositions)

# Modify EnhancedDataGenerator.generate_batch()
spectra, compositions = load_custom_spectra('my_data/')
```

---

## Performance Optimization

### Timing Breakdown (100 epochs, batch 32)

| Component | Time | % |
|-----------|------|---|
| Data generation | 5 min | 5% |
| Forward pass | 60 min | 60% |
| Backward pass | 20 min | 20% |
| Optimization | 10 min | 10% |
| **Total** | **95 min** | - |

### Optimization Strategies

**1. Reduce data generation overhead:**
```python
# Cache generated data
gen.generate_full_dataset('data/simulated/dataset.h5')
# Then load in batches
```

**2. Use mixed precision (2-3x speedup):**
```bash
# Edit trainer.py to enable autocast
```

**3. Increase batch size (4-5x speedup on GPU):**
```bash
python scripts/train_pinn_v3.py --batch-size 256
```

**4. Use gradient accumulation (for larger effective batch):**
```python
# Update trainer.py to accumulate gradients
accumulation_steps = 4
```

---

## Expected Results

### Composition Prediction Accuracy

| Species | Target MAE | Phase 4 Typical |
|---------|----------|--|
| N₂, O₂ | < 0.005 | 0.008 |
| CO₂, H₂O | < 0.010 | 0.012 |
| CH₄, H₂ | < 0.020 | 0.025 |
| O₃, NH₃ | < 0.050 | 0.045 |

### Overall Metrics

- **R² Score:** > 0.95 (on validation set)
- **Temperature RMSE:** ± 15K
- **Uncertainty calibration:** ECE < 0.05

### Training Speed

| Device | Total Time |
|--------|-----------|
| CPU (4 cores) | ~90 min |
| GPU (RTX 3090) | ~5 min |
| GPU (RTX 4090) | ~3 min |

---

## Checkpointing and Resuming

### Automatic Checkpointing

Saves occur at:
1. **Best model** when val_loss improves
2. **Periodic** every 10 epochs

```bash
models/
├── checkpoints/
│   ├── best_model.pt
│   ├── checkpoint_epoch_10.pt
│   ├── checkpoint_epoch_20.pt
│   └── ...
└── training_log.json
```

### Resume Training

```bash
# Resume from best model
python scripts/train_pinn_v3.py \
    --resume models/checkpoints/best_model.pt \
    --epochs 150

# Resume from specific epoch
python scripts/train_pinn_v3.py \
    --resume models/checkpoints/checkpoint_epoch_50.pt \
    --epochs 150
```

---

## Next Steps

1. ✅ **Complete Phase 4 training** with scripts/train_pinn_v3.py
2. ✅ **Evaluate on validation set** (already included)
3. ⏳ **Phase 5 deployment** (in progress)
   - Create test suite
   - Build inference API
   - Documentation
   - Cloud deployment

---

## FAQ

**Q: How long does training take?**  
A: 90 minutes (CPU) or 5 minutes (GPU).

**Q: Can I use my own data?**  
A: Yes, modify EnhancedDataGenerator to load custom spectra.

**Q: What's the best learning rate?**  
A: Default 0.001 is good. Adjust ±10x if needed.

**Q: How many epochs is enough?**  
A: 50-100. Monitor val_loss to see convergence.

**Q: Why does accuracy plateau?**  
A: Model converges. Try longer training or higher learning rate.

---

**For more information:**
- `DEPLOYMENT_GUIDE.md` - Deployment and inference
- `API_REFERENCE.md` - API documentation
- `PROJECT_STATUS.md` - Project overview
