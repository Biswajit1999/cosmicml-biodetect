# Theory: Physics-Informed Neural Networks for Biosignature Detection

This document explains the scientific and technical foundations of CosmicML-Biodetect.

## Table of Contents

1. [Exoplanet Atmospheres & Biosignatures](#exoplanet-atmospheres)
2. [Transmission Spectroscopy](#transmission-spectroscopy)
3. [Neural Networks Basics](#neural-networks)
4. [Physics-Informed Neural Networks (PINNs)](#pinns)
5. [Bayesian Inference](#bayesian-inference)
6. [Atmospheric Chemistry & Radiative Transfer](#atmosphere-physics)

## Exoplanet Atmospheres & Biosignatures {#exoplanet-atmospheres}

### What Makes an Atmosphere Habitable?

A planet's atmosphere determines whether it can support life as we know it. Key factors:

1. **Presence of liquid water** - Requires temperatures between 0-100°C and sufficient pressure
2. **Atmospheric composition** - Must include nitrogen, oxygen, and carbon dioxide in proper ratios
3. **Radiation shield** - Ozone layer protects against UV radiation
4. **Greenhouse effect** - CO2 and H2O keep the planet warm enough for life

### Biosignatures: Chemical Clues to Life

A **biosignature** is a chemical or physical property that indicates biological activity. On Earth:

- **Oxygen (O2)** - 21% of atmosphere, produced entirely by photosynthetic organisms
- **Methane (CH4)** - Rapidly oxidized in our atmosphere; sustained levels indicate biological source
- **Ozone (O3)** - Photochemical product of O2; indicates oxygen-rich atmosphere
- **Complex organics** - Molecules that require biological processes to create

The challenge: Abiotic (non-biological) processes can also produce these molecules under certain conditions.

## Transmission Spectroscopy {#transmission-spectroscopy}

### How We Observe Exoplanet Atmospheres

We cannot directly image exoplanet atmospheres. Instead, we use **transmission spectroscopy**:

1. **Starlight passes through atmosphere** - As a planet transits its host star, light passes through the planet's upper atmosphere
2. **Atmosphere absorbs specific wavelengths** - Different molecules absorb different wavelengths of light
3. **Spectral fingerprint** - The dip in brightness at specific wavelengths reveals atmospheric composition

### The Transmission Spectrum Equation

The observed transit depth is given by:

$$d(\lambda) = \left(\frac{R_p + H}{R_\star}\right)^2 - \left(\frac{R_p}{R_\star}\right)^2$$

Where:
- $R_p$ = planet radius
- $R_\star$ = star radius
- $H(\lambda)$ = scale height (wavelength-dependent due to absorption)
- $\lambda$ = wavelength

The wavelength-dependent scale height contains the atmospheric chemistry information.

### Why This is Hard

- **Small signals**: Transit depth differences are ~0.01-0.1% (need extreme sensitivity)
- **Noise**: Stellar variability, instrumental noise, cloud/haze effects
- **Degeneracies**: Multiple atmospheric compositions can produce similar spectra
- **Limited data**: JWST can observe maybe 10-50 planets well; we need inference power

## Neural Networks Basics {#neural-networks}

### What is a Neural Network?

A neural network is a function $f_\theta(\mathbf{x})$ with parameters $\theta$ that maps input $\mathbf{x}$ to output $\mathbf{y}$:

$$\mathbf{y} = f_\theta(\mathbf{x}) = \sigma(W_n \sigma(W_{n-1} \cdots \sigma(W_1 \mathbf{x} + \mathbf{b}_1) \cdots + \mathbf{b}_{n-1}) + \mathbf{b}_n)$$

Where:
- $W_i$ = weight matrices (learned)
- $\mathbf{b}_i$ = bias vectors (learned)
- $\sigma$ = activation function (ReLU, sigmoid, etc.)

### Training a Neural Network

We minimize a loss function by adjusting parameters:

$$\mathcal{L}(\theta) = \frac{1}{N} \sum_{i=1}^{N} \ell(y_i, f_\theta(\mathbf{x}_i))$$

Using gradient descent:

$$\theta_{k+1} = \theta_k - \alpha \nabla_\theta \mathcal{L}(\theta_k)$$

Where $\alpha$ is the learning rate.

### Why Neural Networks for Spectroscopy?

- Can learn complex non-linear relationships from data
- Automatically extract relevant features
- Can generalize to new data if trained properly
- Fast inference once trained

### The Problem: Neural Networks Are "Black Boxes"

Traditional neural networks have no built-in knowledge of physical constraints. They might:
- Predict negative abundances (physically impossible)
- Violate conservation laws
- Extrapolate poorly beyond training data
- Make predictions that contradict atmospheric physics

**Solution: Physics-Informed Neural Networks (PINNs)**

## Physics-Informed Neural Networks (PINNs) {#pinns}

### The Core Idea

Instead of learning from data alone, we train the network to satisfy BOTH:
1. **Data fidelity**: Predictions match observations
2. **Physics constraints**: Solutions obey the laws of physics

### The PINN Loss Function

The total loss is a weighted combination:

$$\mathcal{L}_{total} = \mathcal{L}_{data} + \lambda \mathcal{L}_{physics}$$

Where:

**Data loss** (Mean Squared Error):
$$\mathcal{L}_{data} = \frac{1}{N_{data}} \sum_{i=1}^{N_{data}} \|f_\theta(\mathbf{x}_i) - \mathbf{y}_i\|^2$$

**Physics loss** (constraint violations):
$$\mathcal{L}_{physics} = \mathcal{L}_{chem} + \mathcal{L}_{thermo} + \mathcal{L}_{conservation}$$

### Physics Constraints for Exoplanet Atmospheres

#### 1. Abundance Constraints
Each species has a mixing ratio $r_i \in [0, 1]$:

$$0 \leq r_i(\mathbf{z}) \leq 1$$

$$\sum_i r_i(\mathbf{z}) = 1$$

Loss term:
$$\mathcal{L}_{cons} = \lambda_{cons} \left[\sum_i (r_i^-)^2 + \left(\sum_i r_i - 1\right)^2\right]$$

Where $r_i^- = \max(0, -r_i)$ enforces non-negativity.

#### 2. Chemical Equilibrium
At thermal equilibrium, the Gibbs free energy is minimized:

$$\sum_j \mu_j n_j = 0 \text{ (for forward reaction)}$$

Where $\mu_j$ is chemical potential and $n_j$ is stoichiometric coefficient.

#### 3. Thermodynamic Consistency
Enthalpy-entropy relationships (Clausius-Clapeyron equation):

$$\frac{d \ln P}{dT} = \frac{\Delta H}{RT^2}$$

### PINN Architecture for Spectroscopy

```
Input Spectrum (512 wavelength bins)
    ↓
Encoder (512 → 256 → 128 → 64 latent dimensions)
    ↓
Physics-Constrained Bottleneck (latent space where physics is enforced)
    ↓
Decoder (64 → 128 → 256 → 32 output abundances)
    ↓
Output: Atmospheric Composition (O2, CH4, N2, H2O, etc.)
```

### Why PINNs Work Better

1. **Better generalization**: Physics constraints prevent unrealistic extrapolations
2. **Less training data needed**: Physics acts as regularization
3. **Interpretability**: Can verify solutions match known physics
4. **Uncertainty quantification**: Physics bounds help define credible regions

## Bayesian Inference {#bayesian-inference}

### Bayes' Theorem

Given observed spectrum $\mathbf{d}$ and atmospheric composition $\mathbf{m}$:

$$P(\mathbf{m} | \mathbf{d}) = \frac{P(\mathbf{d} | \mathbf{m}) \, P(\mathbf{m})}{P(\mathbf{d})}$$

Where:
- $P(\mathbf{m} | \mathbf{d})$ = **Posterior** (what we want: probability of composition given data)
- $P(\mathbf{d} | \mathbf{m})$ = **Likelihood** (how well spectrum matches composition)
- $P(\mathbf{m})$ = **Prior** (prior knowledge about plausible atmospheres)
- $P(\mathbf{d})$ = **Evidence** (normalization constant)

### Why Bayesian for Biosignatures?

1. **Quantifies uncertainty** - Returns probability distributions, not point estimates
2. **Incorporates prior knowledge** - Can include physical constraints, observational biases
3. **Handles degeneracies** - Multiple solutions are represented as probability clouds
4. **Robust to noise** - Bayesian approach naturally smooths noisy data

### Biosignature Probability

Once we have the posterior $P(\mathbf{m} | \mathbf{d})$, we can compute biosignature probabilities:

$$P(\text{O}_2 | \mathbf{d}) = \int_{0.1}^{1} P(\mathbf{m} | \mathbf{d}) \, d m_{O_2}$$

(Integrate over all states where O2 abundance is significant)

## Atmospheric Chemistry & Radiative Transfer {#atmosphere-physics}

### Radiative Transfer Equation

The intensity of light through an atmosphere follows:

$$\frac{dI_\nu}{d\tau_\nu} = -I_\nu + S_\nu$$

Where:
- $I_\nu$ = intensity at frequency $\nu$
- $\tau_\nu$ = optical depth (absorption)
- $S_\nu$ = source function (emission)

For transmission spectroscopy, the optical depth depends on:

$$\tau_\nu = \int n_i(\mathbf{r}) \sigma_i(\nu) d\mathbf{r}$$

Where:
- $n_i$ = number density of species $i$
- $\sigma_i(\nu)$ = absorption cross-section
- Integration is through the atmosphere

### Chemical Network

Species interact through reactions:

$$\mathbf{A}_1 + \mathbf{A}_2 \rightleftharpoons \mathbf{A}_3 + \mathbf{A}_4$$

Rate of change:
$$\frac{d[\mathbf{A}_i]}{dt} = \sum_j k_j [\mathbf{A}_{reactants}] - k_{j,rev} [\mathbf{A}_{products}]$$

Where $k_j$ is the rate constant (temperature-dependent).

### Implementation in CosmicML-Biodetect

We use simplified but physically accurate models:

1. **Pre-computed cross-sections** from molecular databases (HITRAN, ExoMolecules)
2. **Template atmosphere** approach (T, P profiles from atmospheric models)
3. **Fast radiative transfer** using lookup tables
4. **Chemical equilibrium** calculation for selected key reactions

---

## References

### Key Papers on PINNs
- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems. Journal of Computational Physics, 378, 686-707.

### Exoplanet Atmospheres
- Madhusudhan, N., Pearson, K. A., Knutson, H. A., et al. (2014). A high mean molecular weight atmosphere for the super-Earth Kepler-36b. Nature, 469(7328), 64-67.

### Biosignature Detection
- Seager, S., Bains, W., & Petkowski, J. J. (2016). Toward a list of plausible biosignatures for high-resolution exoplanet spectroscopy. Astrobiology, 16(6), 465-485.

### JWST Exoplanet Observations
- Natalie M. Batalha, et al. (2023). Spectroscopic time series of a massive protostellar outflow driven by an O-type star. Nature Astronomy.

---

**Last Updated:** 2024  
**For questions:** See GitHub Discussions
