# CosmicML-Biodetect API Reference

**Version:** 1.0  
**Last Updated:** 2026-06-06

---

## Table of Contents

1. [Inference API](#inference-api)
2. [Training API](#training-api)
3. [Model API](#model-api)
4. [Physics API](#physics-api)

---

## Inference API

### `SpectrumPredictor`

High-level API for spectrum analysis.

```python
from src.cosmicml.inference.predictor import SpectrumPredictor

predictor = SpectrumPredictor(
    model_path: str,
    device: str = 'cpu',
    mc_samples: int = 50,
)
```

**Parameters:**
- `model_path` (str): Path to trained model checkpoint
- `device` (str): 'cpu' or 'cuda'
- `mc_samples` (int): MC dropout samples for uncertainty estimation

**Methods:**

#### `predict(spectrum, return_attention=True) -> PredictionResult`

Predict atmospheric composition from single spectrum.

```python
result = predictor.predict(spectrum)
print(result.temperature)      # float (K)
print(result.composition)      # ndarray [10]
print(result.uncertainty)      # ndarray [10]
print(result.attention_weights) # Dict[str, ndarray]
```

**Parameters:**
- `spectrum` (ndarray or Tensor): [512] or [1, 512] spectrum
- `return_attention` (bool): Return attention weights

**Returns:**
- `PredictionResult`: Contains composition, temperature, uncertainties

---

#### `predict_batch(spectra, batch_size=32) -> List[PredictionResult]`

Process batch of spectra efficiently.

```python
spectra = np.random.randn(100, 512)
results = predictor.predict_batch(spectra, batch_size=32)

temperatures = [r.temperature for r in results]
compositions = np.array([r.composition for r in results])
```

**Parameters:**
- `spectra` (ndarray or Tensor): [N, 512] batch of spectra
- `batch_size` (int): Processing batch size

**Returns:**
- `List[PredictionResult]`: Predictions for each spectrum

---

#### `detect_biosignatures(spectrum, biosignature_species=None, threshold=1e-4) -> Dict[str, bool]`

Check for specific biosignatures.

```python
detections = predictor.detect_biosignatures(
    spectrum,
    biosignature_species=['O3', 'CH4', 'NH3'],
    threshold=1e-4,
)
print(detections['CH4'])  # True if detected, False otherwise
```

**Parameters:**
- `spectrum` (ndarray): [512] spectrum
- `biosignature_species` (List[str]): Species to check (default: ['O3', 'CH4', 'NH3'])
- `threshold` (float): Detection threshold (abundance)

**Returns:**
- `Dict[str, bool]`: Detection status for each species

---

#### `get_species_importance(spectrum) -> Dict[str, float]`

Compute importance score for each species.

```python
importance = predictor.get_species_importance(spectrum)
print(importance['H2O'])  # 0.45 (importance score [0, 1])
```

**Parameters:**
- `spectrum` (ndarray): [512] spectrum

**Returns:**
- `Dict[str, float]`: Species name → importance score

---

#### `explain_prediction(spectrum, top_k=5) -> Dict`

Generate interpretable explanation.

```python
explanation = predictor.explain_prediction(spectrum, top_k=4)
print(explanation['temperature'])  # "300.5 K"
print(explanation['top_species'])
# [
#   {'species': 'N2', 'abundance': 0.68, 'uncertainty': 0.01},
#   {'species': 'O2', 'abundance': 0.21, 'uncertainty': 0.02},
#   ...
# ]
```

**Parameters:**
- `spectrum` (ndarray): [512] spectrum
- `top_k` (int): Number of top species to explain

**Returns:**
- `Dict`: Temperature and top species with abundances

---

### `PredictionResult`

Data class containing prediction results.

**Attributes:**
```python
result.composition       # ndarray [10] species abundances
result.temperature      # float, atmospheric temperature (K)
result.pressure         # Optional[float], surface pressure (bar)
result.uncertainty      # Optional[ndarray [10]], abundance uncertainty
result.aleatoric_unc    # Optional[ndarray [10]], data noise uncertainty
result.epistemic_unc    # Optional[ndarray [10]], model uncertainty
result.attention_weights # Optional[Dict], attention head weights
```

**Methods:**

#### `__str__() -> str`

Pretty-print results.

```python
print(result)
# ============================================================
# SPECTRUM ANALYSIS RESULTS
# ============================================================
# 
# Temperature: 300.0 K
# Pressure: 1.00 bar
# 
# Composition:
# ============================================================
#   N2      : 6.800e-01 ± 1.0e-02
#   O2      : 2.100e-01 ± 2.0e-02
#   ...
```

---

#### `to_dict() -> Dict`

Convert to dictionary.

```python
data = result.to_dict()
# {
#   'composition': [0.68, 0.21, ...],
#   'temperature': 300.0,
#   'pressure': 1.0,
#   'uncertainty': [0.01, 0.02, ...],
# }
```

---

#### `save_json(filepath: str) -> None`

Save results to JSON file.

```python
result.save_json('results/spectrum_1.json')
```

---

### Module Functions

```python
from src.cosmicml.inference.predictor import load_predictor, predict_spectrum

# Load predictor
predictor = load_predictor('models/best_model.pt', device='cpu')

# Single-shot prediction
result = predict_spectrum(spectrum, 'models/best_model.pt')
```

---

## Training API

### `EnhancedDataGenerator`

Physics-based synthetic data generation.

```python
from src.cosmicml.training import EnhancedDataGenerator

gen = EnhancedDataGenerator(
    num_atmospheres: int = 10000,
    wavelength_range: Tuple[float, float] = (0.3, 5.0),
    n_wavelengths: int = 512,
    use_enhanced_rt: bool = True,
)
```

**Parameters:**
- `num_atmospheres` (int): Number of synthetic atmospheres
- `wavelength_range` (Tuple): Min-max wavelength (μm)
- `n_wavelengths` (int): Number of wavelength points
- `use_enhanced_rt` (bool): Use Phase 2 radiative transfer

**Methods:**

#### `generate_batch(batch_size, temperature_range=(200, 500), pressure_range=(0.1, 10))`

Generate batch of spectra.

```python
spectra, compositions, temps, press = gen.generate_batch(
    batch_size=32,
    temperature_range=(200, 500),
    pressure_range=(0.1, 10),
)
# spectra: [32, 512]
# compositions: [32, 10]
# temps: [32]
# press: [32]
```

---

#### `generate_realistic_spectrum(composition, temperature, pressure, star_temp=5778)`

Generate single realistic spectrum.

```python
spectrum = gen.generate_realistic_spectrum(
    composition={'H2O': 0.1, 'CO2': 0.01, ...},
    temperature=300,
    pressure=1.0,
)
# spectrum: [512]
```

---

### `CurriculumSchedule`

Progressive learning curriculum.

```python
from src.cosmicml.training import CurriculumSchedule

curriculum = CurriculumSchedule()
```

**Methods:**

#### `get_current_stage() -> Dict`

Get current curriculum stage.

```python
stage = curriculum.get_current_stage()
# {
#   'name': 'Abundant Species Only',
#   'species_mask': [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
#   'temperature_range': (288, 288),
#   'learning_rate_scale': 1.0,
#   'epochs': 10,
# }
```

---

#### `advance_stage() -> bool`

Move to next curriculum stage.

```python
success = curriculum.advance_stage()
# Returns True if advanced, False if at final stage
```

---

#### `apply_mask_to_composition(composition, mask) -> Tensor`

Apply species mask to composition.

```python
masked_comp = curriculum.apply_mask_to_composition(
    composition,  # [batch, 10]
    stage['species_mask'],
)
# masked_comp: [batch, 10] (renormalized)
```

---

### `TrainingManager`

Complete training orchestration.

```python
from src.cosmicml.training import TrainingManager

trainer = TrainingManager(
    model,
    learning_rate: float = 0.001,
    device: str = 'cpu',
)
```

**Methods:**

#### `train_step(spectra, target_composition, target_temperature) -> Dict[str, float]`

Execute single training step.

```python
metrics = trainer.train_step(
    spectra,              # [batch, 512]
    target_composition,   # [batch, 10]
    target_temperature,   # [batch]
)
# {
#   'total_loss': 0.0042,
#   'data_loss': 0.0038,
#   'temperature_loss': 0.0001,
#   'physics_loss': 0.0003,
#   'learning_rate': 0.001,
# }
```

---

#### `validate(spectra, target_composition) -> Dict[str, float]`

Validation step.

```python
val_metrics = trainer.validate(spectra, target_composition)
# {
#   'val_r2': 0.95,
#   'val_mae_mean': 0.01,
#   'val_mae_rare': 0.05,
# }
```

---

#### `save_checkpoint(epoch: int, val_loss: float) -> None`

Save model checkpoint.

```python
trainer.save_checkpoint(epoch=10, val_loss=0.0042)
# Saves to: models/checkpoints/best_model.pt
#           models/checkpoints/checkpoint_epoch_10.pt (periodic)
```

---

#### `load_checkpoint(checkpoint_path: str) -> int`

Load from checkpoint.

```python
epoch = trainer.load_checkpoint('models/checkpoints/best_model.pt')
```

---

### `AdvancedOptimizer`

AdamW with learning rate scheduling.

```python
from src.cosmicml.training.trainer import AdvancedOptimizer

optimizer = AdvancedOptimizer(
    model,
    learning_rate: float = 0.001,
    weight_decay: float = 0.0001,
    warmup_epochs: int = 10,
    total_epochs: int = 100,
    gradient_clip: Optional[float] = 1.0,
)
```

**Methods:**

#### `step(loss: Tensor) -> None`

Optimizer step with gradient clipping.

```python
optimizer.step(loss)
```

---

#### `get_learning_rate() -> float`

Get current learning rate.

```python
lr = optimizer.get_learning_rate()
print(f"Current LR: {lr:.2e}")
```

---

### `MultiTaskLossBalancer`

Automatic multi-task loss balancing.

```python
from src.cosmicml.training.trainer import MultiTaskLossBalancer

balancer = MultiTaskLossBalancer(
    num_tasks: int = 4,
    initial_weights: Optional[List[float]] = None,
)
```

**Methods:**

#### `compute_weighted_loss(loss_dict, loss_names) -> Tensor`

Compute balanced loss.

```python
loss_dict = {
    'data': torch.tensor(0.5),
    'temperature': torch.tensor(0.1),
    'physics': torch.tensor(0.3),
    'aleatoric': torch.tensor(0.05),
}
total_loss = balancer.compute_weighted_loss(
    loss_dict,
    ['data', 'temperature', 'physics', 'aleatoric']
)
```

---

#### `get_task_weights() -> Tensor`

Get normalized task weights.

```python
weights = balancer.get_task_weights()
# [0.6, 0.1, 0.25, 0.05]
```

---

## Model API

### `PINNv3`

Physics-informed neural network v3.

```python
from src.cosmicml.models.pinn_v3 import PINNv3

model = PINNv3(
    input_dim: int = 512,
    output_dim: int = 10,
    num_attention_heads: int = 4,
    num_mc_samples: int = 10,
    physics_weight: float = 1.0,
    gibbs_weight: float = 0.5,
    equilibrium_weight: float = 0.3,
)
```

**Parameters:**
- `input_dim` (int): Input spectrum dimension
- `output_dim` (int): Number of species (10)
- `num_attention_heads` (int): Attention mechanism heads
- `num_mc_samples` (int): MC dropout samples
- `physics_weight` (float): Physics loss weight
- `gibbs_weight` (float): Gibbs free energy weight
- `equilibrium_weight` (float): Equilibrium constant weight

**Methods:**

#### `forward(spectrum) -> Dict[str, Tensor]`

Forward pass.

```python
outputs = model(spectrum)  # [batch, 512]
# {
#   'composition': [batch, 10],
#   'temperature': [batch, 1],
#   'uncertainty': [batch, 10],
#   'aleatoric_uncertainty': [batch, 10],
#   'attention_weights': [[batch, seq_len], ...],
# }
```

---

#### `compute_physics_loss(composition, temperature) -> Tensor`

Compute physics constraint loss.

```python
physics_loss = model.compute_physics_loss(
    composition,  # [batch, 10]
    temperature,  # [batch]
)
```

---

## Physics API

### `GibbsFreeEnergyCalculator`

Thermodynamic calculations.

```python
from src.cosmicml.physics.thermodynamics import GibbsFreeEnergyCalculator

calc = GibbsFreeEnergyCalculator()
```

**Methods:**

#### `compute_gibbs_energy(composition, temperature) -> Tensor`

Compute ΔG°.

```python
delta_g = calc.compute_gibbs_energy(
    composition,  # [batch, 10]
    temperature,  # [batch]
)
```

---

#### `compute_equilibrium_constant(temperature) -> Dict[str, float]`

Compute K_eq for reactions.

```python
k_eq = calc.compute_equilibrium_constant(temperature=300)
# {
#   'O2+O<->O3': 1.2e-10,
#   'N2+O<->NO': 3.5e-8,
#   ...
# }
```

---

### `EnhancedRadiativeTransfer`

Spectrum simulation with radiative transfer.

```python
from src.cosmicml.physics.radiative_transfer import EnhancedRadiativeTransfer

rt = EnhancedRadiativeTransfer()
```

**Methods:**

#### `compute_optical_depth(wavelength, composition, densities, altitudes, temperatures, pressures)`

Compute optical depth at wavelength.

```python
tau_total, tau_abs, tau_scatter = rt.compute_optical_depth(
    wavelength=1.5,        # μm
    composition={...},     # species abundances
    densities=...,         # [n_layers] m^-3
    altitudes=...,         # [n_layers] km
    temperatures=...,      # [n_layers] K
    pressures=...,         # [n_layers] Pa
)
```

---

## Code Examples

### Basic Inference

```python
from src.cosmicml.inference.predictor import SpectrumPredictor
import numpy as np

# Initialize
predictor = SpectrumPredictor('models/best_model.pt')

# Load spectrum
spectrum = np.loadtxt('spectrum.txt')

# Predict
result = predictor.predict(spectrum)

# Results
print(f"Temperature: {result.temperature:.1f} K")
print(f"N2 abundance: {result.composition[3]:.3e}")
print(f"Uncertainty: ±{result.uncertainty[3]:.1e}")
```

### Training Loop

```python
from src.cosmicml.training import (
    EnhancedDataGenerator, 
    CurriculumSchedule,
    TrainingManager
)
from src.cosmicml.models.pinn_v3 import PINNv3

# Initialize
model = PINNv3(input_dim=512, output_dim=10)
trainer = TrainingManager(model, learning_rate=0.001)
gen = EnhancedDataGenerator()
curriculum = CurriculumSchedule()

# Training loop
for epoch in range(100):
    stage = curriculum.get_current_stage()
    
    # Generate data
    specs, comps, temps, _ = gen.generate_batch(32)
    
    # Apply curriculum
    masked = curriculum.apply_mask_to_composition(comps, stage['mask'])
    
    # Train step
    metrics = trainer.train_step(specs, masked, temps)
    
    # Advance curriculum
    if (epoch + 1) % stage['epochs'] == 0:
        curriculum.advance_stage()
```

### Batch Processing

```python
# Process 1000 spectra
all_spectra = np.random.randn(1000, 512)

results = predictor.predict_batch(all_spectra, batch_size=64)

# Extract data
temps = np.array([r.temperature for r in results])
comp = np.array([r.composition for r in results])

# Statistics
print(f"Mean temperature: {temps.mean():.1f} K")
print(f"Mean composition: {comp.mean(axis=0)}")
```

---

## Error Handling

```python
from src.cosmicml.inference.predictor import SpectrumPredictor
from pathlib import Path

try:
    predictor = SpectrumPredictor('models/best_model.pt')
except FileNotFoundError:
    print("Model not found. Train with: python scripts/train_pinn_v3.py")

try:
    result = predictor.predict(spectrum)
except Exception as e:
    print(f"Prediction error: {e}")
```

---

**For more information, see:**
- `DEPLOYMENT_GUIDE.md` - Deployment instructions
- `TRAINING_GUIDE.md` - Training guide
- `PROJECT_STATUS.md` - Project overview
