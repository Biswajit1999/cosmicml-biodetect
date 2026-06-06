# CosmicML-Biodetect: Deep Research Upgrade Plan

**Status:** In Progress - Major Physics & Architecture Improvements  
**Date:** 2026-06-06  
**Scope:** Complete technical overhaul with real physics implementation

---

## Executive Summary

After comprehensive code review, the current implementation has good structure but weak physics constraints. The PINN claims to enforce physics but the thermodynamic and conservation losses are mostly zero. This upgrade implements real mathematical physics into the model.

---

## Issues Identified

### 1. **PINN Physics Constraints Are Minimal**

**Current State:**
```python
# compute_thermodynamic_loss() - Line 230
thermo_loss = torch.zeros(composition.shape[0], device=composition.device)
return torch.mean(thermo_loss)  # Returns ZERO!
```

**Problem:** This completely ignores thermodynamic constraints.

**Current State:**
```python
# compute_conservation_loss() - Line 193
return torch.tensor(0.0, device=composition.device)  # Always returns ZERO
```

**Problem:** No actual conservation law enforcement.

### 2. **Missing Real Physics Equations**

Currently missing:
- Gibbs free energy minimization (claimed but not implemented)
- Temperature-dependent chemical equilibrium constants
- Constraint from actual reaction network
- Photochemical production/loss rates
- Coupling between radiative transfer and composition

### 3. **Redundant Normalization**

```python
# Line 119-122
composition_positive = self.softplus(composition_logits)  # Ensures positive
composition_normalized = torch.softmax(composition_positive, dim=1)  # Normalizes again
```

**Problem:** Softmax already normalizes. Softplus is redundant and adds non-linearity.

### 4. **Loss Function Issues**

- MSE loss treats all species equally
- Rare species (O3, CH4) have high relative error but low MSE impact
- Log-scale loss would be better for 5+ orders of magnitude variation
- No weighting for physical importance

### 5. **Missing Spectroscopic Physics**

Radiative transfer is simplified:
- No scattering (Rayleigh, Mie)
- No line-by-line treatment
- No collision-induced absorption
- Cross-sections are fixed (not temperature/pressure dependent enough)
- No H2-H2 or H2-He collision-induced absorption

---

## Mathematical Physics to Implement

### 1. **Gibbs Free Energy Minimization**

Current state: Mentioned but not implemented

Proper formulation:
```
G = Σ μᵢ * xᵢ = Σ (ΔGf°ᵢ + RT ln(xᵢ)) * xᵢ

Minimize G subject to:
- Σ xᵢ = 1
- Mass balance: Σ aⱼᵢ * xᵢ = eⱼ (where aⱼᵢ = atoms of element j in species i)
- Physical bounds: 0 ≤ xᵢ ≤ 1
```

ΔGf° temperature dependent:
```
ΔG°(T) = ΔH° - T*ΔS°

where ΔH° and ΔS° from thermodynamic database
```

### 2. **Chemical Equilibrium Constants**

For each reaction:
```
K_eq(T) = exp(-ΔG°(T) / RT)

Must satisfy:
Π(xᵢ^νᵢ) = K_eq(T)

Where νᵢ = stoichiometric coefficients
```

### 3. **Radiative Transfer with Temperature Dependence**

Current: τ = ∫ n(z) σ(λ) dz with σ(T) ∝ √(T_ref/T)

Proper treatment:
```
σ(λ, T, P) = σ₀(λ) * [T/T₀]^α(λ) * line_shape(ν - ν₀, Γ(T,P))

where:
- α(λ) is wavelength-dependent temperature exponent
- Γ(T,P) is pressure-broadened linewidth
- line_shape is Voigt profile (or Lorentz for optically thin)
```

Transit depth with scattering:
```
τ_abs(λ) = ∫ Σᵢ nᵢ(z) σ_abs,i(λ,T,P) dz
τ_scat(λ) = ∫ Σᵢ nᵢ(z) σ_scat,i(λ) dz  [Rayleigh: σ ∝ λ^-4]

τ_total(λ) = τ_abs + τ_scat

Transit depth: d(λ) = (Rp + H_eff(λ))²/R*²  where H_eff depends on τ
```

### 4. **Photochemistry Constraints**

Production/loss for each species:
```
d[Xᵢ]/dt = Pᵢ(λ, [Xⱼ], T, J_λ) - Lᵢ(λ, [Xⱼ], T, J_λ)

Steady state (valid for long timescales):
Pᵢ([Xⱼ]) = Lᵢ([Xⱼ])

where J_λ is photolysis rate (depends on UV flux)
```

---

## Implementation Plan

### Phase 1: Proper Loss Functions (THIS WEEK)

**1.1 Implement Gibbs Free Energy Loss**
- [ ] Create GibbsEnergyModule
- [ ] Temperature-dependent ΔG° calculation
- [ ] Minimize G in loss function (use interior point method)
- [ ] Add to PINN loss computation

**1.2 Implement Chemical Equilibrium Constraints**
- [ ] K_eq calculation from Gibbs energies
- [ ] Reaction quotient check for each reaction
- [ ] Loss for: |log(Q) - log(K_eq)|² for out-of-equilibrium species

**1.3 Replace MSE with Proper Loss**
- [ ] Log-scale loss for rare species: L = Σ log²(xᵢ_pred / xᵢ_true)
- [ ] Weighted MSE for common species
- [ ] Or use Kullback-Leibler divergence (xᵢ as probability distributions)

### Phase 2: Enhanced Physics Constraints (NEXT 2 WEEKS)

**2.1 Radiative Transfer Enhancement**
- [ ] Add Rayleigh scattering term
- [ ] Implement pressure-dependent line broadening
- [ ] Temperature-dependent cross-section function
- [ ] Better spectroscopic data (use HITRAN or similar)

**2.2 Photochemistry Module**
- [ ] Compute production/loss rates from reaction network
- [ ] Photolysis rates (UV dependent)
- [ ] Loss term for photochemical constraints

**2.3 Composition Coupling**
- [ ] PINN output must satisfy chemical equilibrium
- [ ] Use composition as input to chemistry module
- [ ] Compute what spectrum SHOULD be from chemistry
- [ ] Loss includes: |spectrum_computed - spectrum_predicted|

### Phase 3: Model Architecture Improvements (WEEK 3-4)

**3.1 Better Encoder-Decoder**
- [ ] Remove redundant softplus
- [ ] Add attention layers for wavelength features
- [ ] Spectral feature extraction (peaks, slopes, ratios)
- [ ] Separate pathways for common vs rare species

**3.2 Multi-Head Output**
- [ ] Head 1: Abundance predictions
- [ ] Head 2: Temperature/Pressure estimates (from spectrum shape)
- [ ] Head 3: Uncertainty quantification
- [ ] Couple outputs via physics constraints

**3.3 Uncertainty Quantification**
- [ ] Better than MC Dropout
- [ ] Use Bayesian layers with proper priors
- [ ] Output uncertainty in composition
- [ ] Propagate through physics constraints

### Phase 4: Training Improvements (WEEK 5)

**4.1 Better Data**
- [ ] More realistic synthetic atmospheres
- [ ] Real photochemical equilibrium, not random compositions
- [ ] Include observational noise patterns
- [ ] Wavelength-dependent noise

**4.2 Advanced Training**
- [ ] Curriculum learning: Start with simple cases, progress to complex
- [ ] Multi-task learning: auxiliary tasks (T, P prediction)
- [ ] Adversarial losses to prevent unphysical outputs
- [ ] Scheduled loss weighting (physics weight increases over training)

---

## Mathematical Equations to Implement

### Gibbs Free Energy
$$G(T, x_i) = \sum_i \left[ΔG_{f,i}°(T) + RT\ln(x_i)\right] x_i$$

where:
$$ΔG_f°(T) = ΔH_f° - TΔS_f°$$

### Chemical Potential
$$μ_i = μ_i° + RT\ln(a_i) = μ_i° + RT\ln(x_i) \text{ (ideal)}$$

### Equilibrium Constant
$$K_{eq}(T) = \exp\left(-\frac{ΔG°(T)}{RT}\right) = \left(\frac{x_{prod,1}^{ν_1}...}{x_{react,1}^{ν_1}...}\right)_{equilibrium}$$

### Radiative Transfer
$$τ(λ) = \int_0^∞ [σ_{abs}(λ,T,P) + σ_{scat}(λ)] \cdot n(z) \, dz$$

Rayleigh scattering cross-section:
$$σ_{Rayleigh}(λ) = \frac{8π}{3} \left(\frac{2π}{{λ}}\right)^4 α^2$$

### Transit Depth with Wavelength Dependence
$$d(λ) = \frac{[R_p + H_{eff}(λ)]^2 - R_p^2}{R_{*}^2}$$

where effective scale height from absorption:
$$H_{eff}(λ) = H \cdot \sqrt{\frac{τ(λ)}{2}}$$

### Photochemical Balance (Steady State)
$$0 = P_i([X_j], T, J(λ)) - L_i([X_j], T, J(λ))$$

---

## Testing Plan

Each phase will include:
1. **Unit tests** for math correctness
2. **Physics tests** for constraint satisfaction
3. **Synthetic tests** with known solutions
4. **Convergence tests** to ensure training works
5. **Ablation studies** to verify each component helps

Example tests:
- If input is Earth's atmosphere, output should be ~(N2:0.78, O2:0.21, Ar:0.01)
- Temperature perturbation should change equilibrium in correct direction
- Rare species should not exceed physical bounds

---

## Performance Targets

**Current State:**
- R² = -0.6196 (worse than mean)
- MAE = 0.1226

**Target After Upgrade:**
- R² > 0.85 (good prediction)
- MAE < 0.03 (reasonable accuracy)
- Physics constraints violated < 1% of time
- Equilibrium satisfied to within 10% of real systems

---

## Timeline

| Phase | Tasks | Time | Status |
|-------|-------|------|--------|
| 1 | Proper loss functions | 1 week | Starting |
| 2 | Physics constraints | 2 weeks | Planned |
| 3 | Architecture | 2 weeks | Planned |
| 4 | Training & data | 1 week | Planned |
| Validation | Testing & verification | 1 week | Planned |

**Total: ~7 weeks for comprehensive upgrade**

---

## Key References

**Gibbs Free Energy & Thermodynamics:**
- Atkins, P. (2019). Physical Chemistry (11th ed.) - Chapters 5-7
- NIST webbook for thermodynamic data

**Radiative Transfer:**
- Kitzmann et al. (2011) - Radiative transfer in exoplanet atmospheres
- Burrows & Orton (2010) - Spectroscopic studies of transiting exoplanet atmospheres

**Chemical Equilibrium:**
- Liang et al. (2013) - Photochemistry in exoplanet atmospheres
- Zahnle et al. (2006) - Photochemistry in terrestrial exoplanet atmospheres

**Machine Learning for Physics:**
- Raissi et al. (2019) - Physics-Informed Neural Networks
- Han et al. (2018) - High-dimensional PDEs with deep learning

---

## Success Criteria

- ✅ All physics equations properly implemented and tested
- ✅ Model predicts known atmospheres correctly
- ✅ R² > 0.85 on synthetic test set
- ✅ Physics constraints satisfied > 99% of time
- ✅ Rare species (O3, CH4) accurately predicted
- ✅ Uncertainty quantification working properly
- ✅ Code is well-documented with equations
- ✅ All tests passing (>80% coverage)

---

This is a real research upgrade, not cosmetic improvements. Every change adds actual physics.
