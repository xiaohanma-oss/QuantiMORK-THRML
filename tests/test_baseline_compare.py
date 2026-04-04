"""Regression tests comparing WaveletPCTransformer against PC-Transformers baseline.

Verifies the core claim: wavelet-sparse MLP produces comparable learning
quality to dense MLP, measured by CE loss and PC energy on Tiny Shakespeare.
"""

import math
import os
import sys

import pytest
import torch
import torch.nn.functional as F

# Add vendor to path (same pattern as test_model.py)
VENDOR_DIR = os.path.join(os.path.dirname(__file__), "..", "vendor", "PC-Transformers")
sys.path.insert(0, VENDOR_DIR)

from predictive_coding.pc_layer import PCLayer

# Shakespeare encoded data path
_DATA_DIR = os.path.join(VENDOR_DIR, "data_preparation", "encoded")
_HAS_DATA = os.path.exists(os.path.join(_DATA_DIR, "valid.pt"))


class _CompareConfig:
    """Config compatible with both PCTransformer and WaveletPCTransformer."""
    vocab_size = 1024
    block_size = 64
    n_embed = 128
    num_heads = 4
    n_blocks = 2
    T = 2
    dropout = 0.0
    lr = 0.001
    peak_learning_rate = 0.001
    warmup_steps = 5
    update_bias = False
    internal_energy_fn_name = "pc_e"
    output_energy_fn_name = "pc_e"
    combined_internal_weight = 0.8
    combined_output_weight = 0.2
    use_flash_attention = False
    alpha = 0.5
    wavelet_n_levels = 3
    batch_size = 8
    num_epochs = 1


def _build_models(config, seed=42):
    """Construct both models with the same random seed."""
    from model_architecture.pc_t_model import PCTransformer
    from quantimork_thrml.model import WaveletPCTransformer

    torch.manual_seed(seed)
    baseline = PCTransformer(config)

    torch.manual_seed(seed)
    wavelet = WaveletPCTransformer(config)

    return baseline, wavelet


def _collect_energy(model):
    """Collect PC energies from all PCLayers (same pattern as train.py)."""
    energies = []
    for m in model.modules():
        if isinstance(m, PCLayer) and hasattr(m, "get_energy"):
            e = m.get_energy()
            if e is not None and not (isinstance(e, float) and math.isnan(e)):
                energies.append(e)
    return energies


def _ce_loss(logits, target_ids):
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        target_ids.view(-1),
        ignore_index=0,
    ).item()


# ---------------------------------------------------------------------------
# 1. Structural comparison (fast, synthetic data)
# ---------------------------------------------------------------------------

class TestStructuralCompare:
    """Fast structural checks — no Shakespeare data needed."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.config = _CompareConfig()
        self.baseline, self.wavelet = _build_models(self.config)
        torch.manual_seed(0)
        B, S = 4, self.config.block_size
        self.input_ids = torch.randint(0, self.config.vocab_size, (B, S))
        self.target_ids = torch.randint(0, self.config.vocab_size, (B, S))

    def test_output_shapes_match(self):
        logits_b = self.baseline(self.target_ids, self.input_ids)
        logits_w = self.wavelet(self.target_ids, self.input_ids)
        assert logits_b.shape == logits_w.shape

    def test_both_finite(self):
        logits_b = self.baseline(self.target_ids, self.input_ids)
        logits_w = self.wavelet(self.target_ids, self.input_ids)
        assert torch.isfinite(logits_b).all()
        assert torch.isfinite(logits_w).all()

    def test_energy_collected(self):
        self.baseline(self.target_ids, self.input_ids)
        self.wavelet(self.target_ids, self.input_ids)
        e_b = _collect_energy(self.baseline)
        e_w = _collect_energy(self.wavelet)
        assert len(e_b) > 0, "Baseline produced no energy"
        assert len(e_w) > 0, "Wavelet produced no energy"

    def test_wavelet_fewer_params(self):
        p_b = sum(p.numel() for p in self.baseline.parameters())
        p_w = sum(p.numel() for p in self.wavelet.parameters())
        assert p_w < p_b

    def test_param_compression_ratio(self):
        p_b = sum(p.numel() for p in self.baseline.parameters())
        p_w = sum(p.numel() for p in self.wavelet.parameters())
        ratio = p_b / p_w
        assert 1.5 <= ratio <= 2.5, (
            f"Compression ratio {ratio:.2f} outside [1.5, 2.5]")


# ---------------------------------------------------------------------------
# 2. Shakespeare training comparison (core regression test)
# ---------------------------------------------------------------------------

_skip_no_data = pytest.mark.skipif(
    not _HAS_DATA, reason="Shakespeare encoded data not found")


@pytest.mark.slow
@_skip_no_data
class TestShakespeareCompare:
    """Core test: train both models on Shakespeare and compare CE loss."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.config = _CompareConfig()
        self.n_steps = 20

    def _train_steps(self, model, loader, n_steps):
        """Run n forward passes (each is a Hebbian training step)."""
        model.train()
        losses = []
        energies_history = []
        step = 0
        for batch in loader:
            if step >= n_steps:
                break
            input_ids = batch["input_ids"]
            target_ids = batch["target_ids"]
            if target_ids.max() >= self.config.vocab_size:
                target_ids = torch.clamp(target_ids, max=self.config.vocab_size - 1)

            # Set learning rate on PC layers
            for m in model.modules():
                if hasattr(m, "local_lr"):
                    m.set_learning_rate(self.config.lr)

            logits = model(target_ids, input_ids)
            ce = _ce_loss(logits, target_ids)
            losses.append(ce)
            energies_history.append(_collect_energy(model))
            step += 1
        return losses, energies_history

    def test_both_learn(self):
        """Both models' CE loss should decrease over training steps."""
        from data_preparation.dataloader import get_loaders
        train_loader, _, _ = get_loaders(distributed=False)

        baseline, wavelet = _build_models(self.config)

        losses_b, _ = self._train_steps(baseline, train_loader, self.n_steps)
        losses_w, _ = self._train_steps(wavelet, train_loader, self.n_steps)

        # Average first 3 vs last 3 steps
        init_b = sum(losses_b[:3]) / 3
        final_b = sum(losses_b[-3:]) / 3
        init_w = sum(losses_w[:3]) / 3
        final_w = sum(losses_w[-3:]) / 3

        assert final_b < init_b, (
            f"Baseline not learning: init CE {init_b:.4f} → final {final_b:.4f}")
        assert final_w < init_w, (
            f"Wavelet not learning: init CE {init_w:.4f} → final {final_w:.4f}")

    def test_ce_gap_bounded(self):
        """Wavelet CE loss should be within 5% of baseline after training."""
        from data_preparation.dataloader import get_loaders
        train_loader, _, _ = get_loaders(distributed=False)

        baseline, wavelet = _build_models(self.config)

        losses_b, _ = self._train_steps(baseline, train_loader, self.n_steps)
        losses_w, _ = self._train_steps(wavelet, train_loader, self.n_steps)

        # Compare final CE (average of last 3 steps for stability)
        final_b = sum(losses_b[-3:]) / 3
        final_w = sum(losses_w[-3:]) / 3
        gap = abs(final_w - final_b) / final_b

        assert gap < 0.05, (
            f"CE gap {gap:.1%} exceeds 5%: "
            f"baseline={final_b:.4f}, wavelet={final_w:.4f}")

    def test_energy_ratio_bounded(self):
        """PC energy ratio between models should be bounded."""
        from data_preparation.dataloader import get_loaders
        train_loader, _, _ = get_loaders(distributed=False)

        baseline, wavelet = _build_models(self.config)

        _, energies_b = self._train_steps(baseline, train_loader, 5)
        _, energies_w = self._train_steps(wavelet, train_loader, 5)

        # Use last step's total energy
        total_b = sum(energies_b[-1]) if energies_b[-1] else 1.0
        total_w = sum(energies_w[-1]) if energies_w[-1] else 1.0
        ratio = total_w / total_b

        assert 0.3 <= ratio <= 3.0, (
            f"Energy ratio {ratio:.2f} outside [0.3, 3.0]")


# ---------------------------------------------------------------------------
# 3. Shakespeare validation PPL (eval-only, no training)
# ---------------------------------------------------------------------------

@pytest.mark.slow
@_skip_no_data
class TestShakespeareValPPL:
    """Eval-only: both models on Shakespeare validation data."""

    def test_val_ce_same_order(self):
        """Random-init models should produce similar CE on real data."""
        from data_preparation.dataloader import get_loaders
        _, val_loader, _ = get_loaders(distributed=False)

        config = _CompareConfig()
        baseline, wavelet = _build_models(config)
        baseline.eval()
        wavelet.eval()

        ces_b, ces_w = [], []
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= 5:
                    break
                input_ids = batch["input_ids"]
                target_ids = batch["target_ids"]
                if target_ids.max() >= config.vocab_size:
                    target_ids = torch.clamp(target_ids, max=config.vocab_size - 1)

                logits_b = baseline(target_ids, input_ids)
                logits_w = wavelet(target_ids, input_ids)
                ces_b.append(_ce_loss(logits_b, target_ids))
                ces_w.append(_ce_loss(logits_w, target_ids))

        avg_b = sum(ces_b) / len(ces_b)
        avg_w = sum(ces_w) / len(ces_w)
        gap = abs(avg_w - avg_b) / avg_b

        assert gap < 0.50, (
            f"Val CE gap {gap:.1%} exceeds 50%: "
            f"baseline={avg_b:.4f}, wavelet={avg_w:.4f}")
