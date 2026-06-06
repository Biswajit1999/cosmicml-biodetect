# Contributing to CosmicML-Biodetect

Thank you for your interest in contributing to this groundbreaking astrobiology project! We welcome contributions from researchers, ML engineers, students, and astronomy enthusiasts.

## Ways to Contribute

### 1. Code Contributions
- **Extend the atmospheric simulator** with new chemistry reactions or radiative transfer models
- **Improve the PINN architecture** with novel loss functions or regularization techniques
- **Add real data pipelines** for JWST, Keck, or other telescopes
- **Optimize GPU performance** for faster training on large datasets
- **Add new inference methods** beyond Bayesian approaches

### 2. Research & Science
- **Validate against real exoplanet observations** from the literature
- **Extend to new biosignature molecules** (phosphine, dimethyl sulfide, etc.)
- **Develop new physical constraints** for the PINN loss function
- **Compare with other detection methods** in the literature
- **Publish papers** using this framework

### 3. Documentation & Education
- **Create tutorials** for newcomers to PINNs or exoplanet science
- **Write documentation** for specific modules
- **Add example notebooks** showing specific use cases
- **Translate materials** to other languages
- **Improve explanations** of complex concepts

### 4. Testing & Quality Assurance
- **Write unit tests** for new modules
- **Test on different hardware** (GPU models, clusters)
- **Report bugs** with reproducible examples
- **Suggest performance improvements**
- **Test with real data** from JWST and other observatories

### 5. Community Building
- **Discuss ideas** in GitHub Issues
- **Share results** in Discussions
- **Mentor new contributors**
- **Connect with other astronomy/ML projects**

## Getting Started

### Setting up Development Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/cosmicml-biodetect.git
cd cosmicml-biodetect

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black src/ tests/
```

### Development Workflow

1. **Create a branch** for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the style guide below

3. **Write/update tests** for your code:
   ```bash
   pytest tests/test_your_module.py -v
   ```

4. **Format your code**:
   ```bash
   black src/ tests/
   flake8 src/ tests/
   ```

5. **Commit with clear messages**:
   ```bash
   git commit -m "Add PINN loss function for chemical equilibrium constraints"
   ```

6. **Push and create a Pull Request**

## Code Style Guide

### Python Style
- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with line length of 100 characters
- Use type hints for all function arguments and returns
- Write docstrings in Google style format
- Use clear, descriptive variable names

### Example
```python
def compute_transit_depth(
    wavelength: np.ndarray,
    atmosphere_composition: Dict[str, float],
    planet_radius: float,
    star_radius: float,
) -> np.ndarray:
    """
    Compute transmission spectrum transit depth.

    Calculates the wavelength-dependent transit depth from atmospheric
    absorption using radiative transfer theory.

    Args:
        wavelength: Array of wavelengths in micrometers
        atmosphere_composition: Dict mapping molecule names to mixing ratios
        planet_radius: Planet radius in Earth radii
        star_radius: Star radius in Solar radii

    Returns:
        Transit depth as function of wavelength (unitless)
    """
    # Implementation here
    pass
```

### Documentation
- Add docstrings to all functions and classes
- Include mathematical equations where relevant
- Provide example usage in docstrings
- Link to relevant papers in the literature

## Testing Requirements

All new code must include tests:

```python
import pytest
from cosmicml.models import PINN

def test_pinn_forward_pass():
    """Test PINN forward pass with synthetic data."""
    pinn = PINN(input_dim=512, latent_dim=64, output_dim=32)
    spectrum = torch.randn(4, 512)  # batch of 4 spectra
    output = pinn.forward(spectrum)
    
    assert output.shape == (4, 32)
    assert not torch.any(torch.isnan(output))
```

## Commit Message Guidelines

Write clear commit messages describing what and why:

**Good:**
```
Add PINN physics loss for chemical equilibrium constraints

- Implement loss function enforcing Gibbs free energy minimization
- Add test cases for various atmospheric compositions
- Update documentation with new constraint details

Fixes #42
```

**Avoid:**
```
Fix bug
Update code
Minor changes
```

## Pull Request Process

1. **Update documentation** if behavior changes
2. **Add tests** covering your changes
3. **Run full test suite**: `pytest tests/ -v`
4. **Check coverage**: `pytest --cov=src tests/`
5. **Format code**: `black src/ tests/`
6. **Write descriptive PR title and description**

### PR Description Template
```markdown
## Description
Brief explanation of the changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation
- [ ] Performance improvement

## Related Issues
Closes #123

## Testing
- [ ] Added unit tests
- [ ] Tested on GPU
- [ ] Tested with real data
- [ ] Verified no regressions

## Checklist
- [ ] Code follows style guidelines
- [ ] Documentation is updated
- [ ] All tests pass
```

## Major Feature Development

For substantial new features:

1. **Open an issue first** to discuss the approach
2. **Get feedback** from maintainers
3. **Create a design document** if complex
4. **Implement incrementally** with regular reviews
5. **Update documentation** throughout development

## Research & Publication

If you publish research using this code:

1. **Cite the repository**: See README.md for citation format
2. **Acknowledge contributors**: List team members
3. **Share preprints**: Post on arXiv
4. **Link to the code**: Include GitHub URL
5. **Provide datasets**: Share preprocessed data if possible

## Questions?

- **Issues**: Create a GitHub Issue for bugs or feature requests
- **Discussions**: Start a Discussion for questions
- **Email**: Contact the maintainers

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/).

We are committed to providing a welcoming and inspiring community for all people, regardless of background or identity. We expect all community members to:

- Be respectful and inclusive
- Welcome diverse perspectives
- Focus on what is best for the community
- Report inappropriate behavior to the maintainers

---

**Thank you for contributing to the search for life in the universe!** 🌍🔭
