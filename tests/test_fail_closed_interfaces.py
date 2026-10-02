import numpy as np
import pytest

from cosmicml.data.jwst_pipeline import JWSTDataPipeline
from cosmicml.data.loader import DataLoader
from cosmicml.inference.bayesian import BayesianInference


def test_jwst_interfaces_do_not_fabricate_observations(tmp_path):
    with pytest.raises(NotImplementedError, match="not implemented"):
        DataLoader(str(tmp_path)).load_jwst_data("example")
    with pytest.raises(NotImplementedError, match="not implemented"):
        JWSTDataPipeline().load_jwst_spectrum("missing.fits")


def test_bayesian_interface_does_not_return_zero_posterior():
    inference = BayesianInference(pinn_model=None, likelihood=None)
    with pytest.raises(NotImplementedError, match="not been implemented"):
        inference.compute_posterior(np.ones(8), np.ones(8) * 0.1)
