"""TSU factor graph energy equivalence tests.

These tests require thrml + JAX. Run with PLN-THRML's venv:
    /path/to/PLN-THRML/.venv/bin/python -m pytest tests/test_thrml_verify.py -v

Or skip if thrml is not installed.
"""

import pytest
import numpy as np

try:
    # Import directly from the module file to avoid __init__.py's torch dependency
    import importlib.util
    import os
    _spec = importlib.util.spec_from_file_location(
        "thrml_verify",
        os.path.join(os.path.dirname(__file__), "..", "quantimork_thrml", "thrml_verify.py"))
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    build_single_level_graph = _mod.build_single_level_graph
    run_verification = _mod.run_verification
    pc_prediction_weights = _mod.pc_prediction_weights
    pc_prior_weights = _mod.pc_prior_weights
    td_modulation_weights = _mod.td_modulation_weights
    gaussian_prior_probs = _mod.gaussian_prior_probs
    kl_prior_weights = _mod.kl_prior_weights
    coeff_bin_centers = _mod.coeff_bin_centers
    HAS_THRML = True
except (ImportError, ModuleNotFoundError):
    HAS_THRML = False

pytestmark = pytest.mark.skipif(not HAS_THRML, reason="thrml not installed")


class TestWeightComputation:
    def test_prediction_weights_shape(self):
        W = pc_prediction_weights(1.0, k=16)
        assert W.shape == (16, 16)

    def test_prediction_weights_diagonal_max(self):
        W = pc_prediction_weights(1.0, k=16)
        for i in range(16):
            assert W[i, i] >= W[i, :].max() - 1e-6

    def test_prior_weights_shape(self):
        W = pc_prior_weights(0.0, 1.0, k=16)
        assert W.shape == (16,)

    def test_bin_centers_count(self):
        centers = coeff_bin_centers(k=16, lo=-1.0, hi=1.0)
        assert len(centers) == 16


class TestFactorGraphVerification:
    def test_small_linear_recovery(self):
        """Core test: factor graph Gibbs sampling recovers approximate target."""
        np.random.seed(42)
        W = np.random.randn(8, 8).astype(np.float32) * 0.1
        x_in = np.random.randn(8).astype(np.float32) * 0.5
        target = (W @ x_in).astype(np.float32)

        graph = build_single_level_graph(W, x_in, target, precision=2.0, k=16)
        result = run_verification(graph, seed=42)

        assert result["mse"] < 0.10, f"MSE {result['mse']:.4f} > 0.10"

    def test_identity_weight_recovery(self):
        """Identity weight → output should match input closely."""
        W = np.eye(8, dtype=np.float32) * 0.5
        x_in = np.array([0.1, 0.2, -0.1, 0.3, -0.2, 0.0, 0.15, -0.05],
                        dtype=np.float32)
        target = (W @ x_in).astype(np.float32)

        graph = build_single_level_graph(W, x_in, target, precision=3.0, k=16)
        result = run_verification(graph, seed=123)

        assert result["mse"] < 0.10

    def test_different_seeds_converge(self):
        """Different random seeds should give similar results."""
        np.random.seed(42)
        W = np.random.randn(8, 8).astype(np.float32) * 0.1
        x_in = np.random.randn(8).astype(np.float32) * 0.5
        target = (W @ x_in).astype(np.float32)

        graph = build_single_level_graph(W, x_in, target, precision=2.0, k=16)

        r1 = run_verification(graph, seed=0)
        r2 = run_verification(graph, seed=99)

        # Both should pass threshold
        assert r1["mse"] < 0.15
        assert r2["mse"] < 0.15


class TestTopDownModulation:
    def test_td_modulation_weights_shape(self):
        W = td_modulation_weights(0.5, 1.0, k=16)
        assert W.shape == (16, 16)

    def test_td_modulation_weights_diagonal_max(self):
        """Diagonal should dominate (matching td = low energy)."""
        W = td_modulation_weights(0.5, 1.0, k=16)
        for i in range(16):
            assert W[i, i] >= W[i, :].max() - 1e-6

    def test_td_graph_still_recovers(self):
        """Factor graph with td modulation should still recover targets."""
        np.random.seed(42)
        W = np.random.randn(8, 8).astype(np.float32) * 0.1
        x_in = np.random.randn(8).astype(np.float32) * 0.5
        target = (W @ x_in).astype(np.float32)
        # td_activations close to target (simulating good top-down prediction)
        td = target + np.random.randn(8).astype(np.float32) * 0.05

        graph = build_single_level_graph(
            W, x_in, target, precision=2.0, k=16,
            td_activations=td, td_alpha=0.5)
        result = run_verification(graph, seed=42)

        assert result["mse"] < 0.15, f"MSE {result['mse']:.4f} > 0.15"


class TestKLDivergence:
    def test_gaussian_prior_probs_sums_to_one(self):
        probs = gaussian_prior_probs(0.0, 1.0, k=16)
        assert abs(float(probs.sum()) - 1.0) < 1e-5

    def test_kl_prior_weights_shape(self):
        probs = gaussian_prior_probs(0.0, 1.0, k=16)
        W = kl_prior_weights(probs, beta=0.1, k=16)
        assert W.shape == (16,)

    def test_kl_prior_weights_peak_at_mode(self):
        """KL prior weight should be highest near the prior mode."""
        import jax.numpy as jnp
        probs = gaussian_prior_probs(0.0, 0.5, k=16, lo=-1.0, hi=1.0)
        W = kl_prior_weights(probs, beta=0.1, k=16)
        centers = coeff_bin_centers(k=16, lo=-1.0, hi=1.0)
        center_bin = int(jnp.argmin(jnp.abs(centers)))
        assert W[center_bin] >= W.max() - 1e-6

    def test_graph_with_kl_recovers(self):
        """Factor graph with KL prior should still recover approximate targets."""
        np.random.seed(42)
        W = np.random.randn(8, 8).astype(np.float32) * 0.1
        x_in = np.random.randn(8).astype(np.float32) * 0.5
        target = (W @ x_in).astype(np.float32)

        graph = build_single_level_graph(
            W, x_in, target, precision=2.0, k=16, beta=0.1)
        result = run_verification(graph, seed=42)

        assert result["mse"] < 0.15, f"MSE {result['mse']:.4f} > 0.15"
