"""
WaveletPCTransformer — PC-Transformers with wavelet-sparse MLP.

Replaces the dense MLP block (fc1: 512→2048, fc2: 2048→512) with a single
WaveletLinear (512→512), implementing whitepaper §7.4.2 + §7.4.3.

The rest of the architecture (Embedding, Attention, PCLayer, Output) is
imported directly from vendor/PC-Transformers.
"""

import sys
import os
import torch
import torch.nn as nn

# Add vendor to Python path for imports
_vendor_dir = os.path.join(os.path.dirname(__file__), "..", "vendor", "PC-Transformers")
if _vendor_dir not in sys.path:
    sys.path.insert(0, os.path.abspath(_vendor_dir))

from predictive_coding.pc_layer import PCLayer
from model_architecture.embedding import Embedding_Layer
from model_architecture.attention import Attention
from model_architecture.output import OutputLayer
from utils.pc_utils import ids_to_one_hot
from utils.device_utils import (
    create_streams_or_futures,
    execute_parallel,
    synchronize_execution,
)

from quantimork_thrml.wavelet_linear import WaveletLinear
from quantimork_thrml.haar import haar_dwt_1d


class WaveletMLP(nn.Module):
    """Wavelet-sparse MLP replacing dense fc1+fc2.

    Dense MLP:  fc1(512→2048) → GELU → dropout → fc2(2048→512)
                = 2 × dense matmul, 262K + 1049K = 1.3M params

    Wavelet MLP: WaveletLinear(512→512, 3 levels)
                 = Haar DWT → per-level small Linear → IDWT
                 = ~90K params, max 5 connections/node
    """

    def __init__(self, config):
        super().__init__()
        n_levels = getattr(config, "wavelet_n_levels", 3)
        self.wavelet = WaveletLinear(
            config.n_embed, config.n_embed, n_levels=n_levels)
        self.dropout = nn.Dropout(config.dropout)

        self.pc_layer1 = PCLayer(
            T=config.T,
            lr=config.lr,
            update_bias=config.update_bias,
            energy_fn_name=config.internal_energy_fn_name,
        )
        # pc_layer2 is kept for interface compatibility with PCTransformer's
        # forward loop, but delegates to the same wavelet layer.
        self.pc_layer2 = PCLayer(
            T=config.T,
            lr=config.lr,
            update_bias=config.update_bias,
            energy_fn_name=config.internal_energy_fn_name,
        )

        # Expose a "fc1" and "fc2" interface for step_linear compatibility.
        # step_linear accesses layer.weight for error projection.
        # We use thin wrapper layers that have .weight attributes.
        self.fc1 = _WaveletLinearProxy(self.wavelet, "forward")
        self.fc2 = _IdentityWithWeight(config.n_embed)


class _WaveletLinearProxy(nn.Module):
    """Proxy that exposes WaveletLinear as a step_linear-compatible layer.

    step_linear needs:
      - layer(x) → mu  (forward pass)
      - layer.weight    (for error projection: bu_err @ layer.weight)

    Weight updates: step_linear applies Hebbian delta to layer.weight, but
    WaveletLinear's actual weights are per-level small matrices. We let
    step_linear write to a disposable identity buffer, then apply the correct
    wavelet-domain Hebbian update via apply_hebbian_update().
    """

    def __init__(self, wavelet_linear, mode="forward"):
        super().__init__()
        self.wavelet_linear = wavelet_linear
        n = wavelet_linear.in_features
        self.register_buffer(
            "_proxy_weight", torch.eye(n))
        self._cached_input = None

    @property
    def weight(self):
        """Identity proxy for step_linear error projection.

        Error projection bu_err @ I = bu_err passes errors through unchanged.
        This is valid because WaveletLinear preserves dimensionality (in == out).
        step_linear also writes Hebbian delta here, which we discard — the real
        update happens in apply_hebbian_update().
        """
        return self._proxy_weight

    @property
    def bias(self):
        return None

    def forward(self, x):
        self._cached_input = x.detach()
        return self.wavelet_linear(x)

    def apply_hebbian_update(self, bu_err, local_lr, td_err=None, td_alpha=0.5,
                             beta=0.0):
        """Apply wavelet-domain Hebbian update to per-level weights.

        Since Haar DWT is orthogonal, the full-space Hebbian delta
        delta_W = lr * (bu_err^T @ x_input) decomposes exactly:
            delta_level_i = lr * DWT(bu_err)_i^T @ DWT(x_input)_i

        When td_err is provided, the combined error at each wavelet level is:
            combined_i = bu_err_i + td_alpha * td_err_i
        This implements bidirectional free energy: bottom-up prediction error
        plus top-down modulation from the layer above.

        When beta > 0, a KL regularization penalty (weight decay toward zero)
        is applied: for Gaussian prior N(0, σ²), ∇_w D_KL ∝ w, so the
        penalty is simply beta * w.

        Args:
            bu_err: Bottom-up prediction error from PCLayer, shape (B, S, D).
            local_lr: PC local learning rate.
            td_err: Top-down error from pc_layer2, shape (B, S, D), or None.
            td_alpha: Weight for top-down error contribution (default 0.5).
            beta: KL divergence regularization strength (default 0.0).
        """
        if self._cached_input is None:
            return

        n_levels = self.wavelet_linear.n_levels
        wavelet_input = haar_dwt_1d(self._cached_input, n_levels)
        wavelet_err = haar_dwt_1d(bu_err, n_levels)

        # Combine bu_err with td_err in wavelet domain
        if td_err is not None and td_alpha > 0:
            wavelet_td = haar_dwt_1d(td_err, n_levels)
            combined_details = [
                wavelet_err["details"][i] + td_alpha * wavelet_td["details"][i]
                for i in range(n_levels)
            ]
            combined_approx = (
                wavelet_err["approx"] + td_alpha * wavelet_td["approx"])
        else:
            combined_details = wavelet_err["details"]
            combined_approx = wavelet_err["approx"]

        with torch.no_grad():
            for i, dt in enumerate(self.wavelet_linear.detail_transforms):
                delta = local_lr * torch.einsum(
                    "bsv, bsh -> vh",
                    combined_details[i],
                    wavelet_input["details"][i])
                delta = torch.clamp(delta, -0.01, 0.01)
                dt.weight.data.add_(delta)
                if dt.bias is not None:
                    delta_b = local_lr * combined_details[i].mean(
                        dim=(0, 1))
                    delta_b = torch.clamp(delta_b, -0.01, 0.01)
                    dt.bias.data.add_(delta_b)
                # KL regularization: ∇_w D_KL = beta * w (Gaussian prior)
                if beta > 0:
                    kl_penalty = torch.clamp(beta * dt.weight.data, -0.01, 0.01)
                    dt.weight.data.add_(-local_lr * kl_penalty)

            at = self.wavelet_linear.approx_transform
            delta_a = local_lr * torch.einsum(
                "bsv, bsh -> vh",
                combined_approx,
                wavelet_input["approx"])
            delta_a = torch.clamp(delta_a, -0.01, 0.01)
            at.weight.data.add_(delta_a)
            if at.bias is not None:
                delta_b = local_lr * combined_approx.mean(dim=(0, 1))
                delta_b = torch.clamp(delta_b, -0.01, 0.01)
                at.bias.data.add_(delta_b)
            # KL regularization for approx level
            if beta > 0:
                kl_penalty = torch.clamp(beta * at.weight.data, -0.01, 0.01)
                at.weight.data.add_(-local_lr * kl_penalty)

        # Reset proxy weight (discard step_linear's spurious delta)
        self._proxy_weight.copy_(torch.eye(
            self._proxy_weight.shape[0], device=self._proxy_weight.device))
        self._cached_input = None


class _IdentityWithWeight(nn.Module):
    """Identity layer with a .weight attribute for step_linear compatibility.

    In the wavelet MLP, fc2's role (2048→512 contraction) is absorbed into
    the single WaveletLinear. This identity layer satisfies the interface.
    """

    def __init__(self, n_features):
        super().__init__()
        self.register_buffer(
            "weight",
            torch.eye(n_features),
        )
        self.bias = None

    def forward(self, x):
        return x


class WaveletTransformerBlock(nn.Module):
    """Transformer block with wavelet MLP (attention unchanged)."""

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.RMSNorm(config.n_embed)
        self.attn = Attention(config)
        self.ln2 = nn.RMSNorm(config.n_embed)
        self.mlp = WaveletMLP(config)


class WaveletPCTransformer(nn.Module):
    """PCTransformer with wavelet-sparse MLP blocks.

    Architecture identical to PC-Transformers except:
    - MLP blocks use WaveletLinear instead of dense fc1+fc2
    - Everything else (Embedding, Attention, PCLayer, Output) is unchanged
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self._td_alpha = getattr(config, "td_alpha", 0.5)
        self._beta = getattr(config, "beta", 0.0)
        self.embedding = Embedding_Layer(config)
        self.blocks = nn.ModuleList(
            [WaveletTransformerBlock(config) for _ in range(config.n_blocks)])
        self.output = OutputLayer(config)

    def register_all_lateral_weights(self):
        """Register lateral weights for all PC layers."""
        for block in self.blocks:
            block.attn.pc_qkv.register_lateral(
                "attn", block.attn.q.in_features)
            block.attn.pc_output.register_lateral(
                "linear", block.attn.output.in_features)
            block.mlp.pc_layer1.register_lateral(
                "fc1", block.mlp.wavelet.in_features)
            block.mlp.pc_layer2.register_lateral(
                "linear", block.mlp.wavelet.in_features)
        self.output.pc_layer.register_lateral(
            "linear", self.output.output.in_features)

        for module in self.modules():
            if hasattr(module, "W_latents"):
                for key in module.W_latents:
                    if module.W_latents[key] is not None:
                        module.W_latents[key] = module.W_latents[key].to(
                            next(self.parameters()).device)

    def forward(self, target_ids, input_ids, use_kv_cache=False):
        """Forward pass — identical logic to PCTransformer.forward."""
        for module in self.modules():
            if hasattr(module, "clear_energy"):
                module.clear_energy()
            if hasattr(module, "clear_errors"):
                module.clear_errors()

        B, S = input_ids.shape
        device = input_ids.device
        vocab_size = self.output.config.vocab_size

        if input_ids.max() >= vocab_size:
            input_ids = torch.clamp(input_ids, max=vocab_size - 1)
        if target_ids.max() >= vocab_size:
            target_ids = torch.clamp(target_ids, max=vocab_size - 1)

        target_logits = ids_to_one_hot(target_ids, vocab_size).to(device)
        position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, S)

        # Initialize all PC layers
        self.embedding.pc_layer.init_x(
            batch_size=B, seq_len=S, layer_type="embed", device=device,
            layer={"word": self.embedding.word_embeddings,
                   "pos": self.embedding.position_embeddings},
            proj_layers=None, input_ids=input_ids, position_ids=position_ids)

        for block in self.blocks:
            block.attn.pc_qkv.init_x(
                batch_size=B, seq_len=S, layer_type="attn", device=device,
                layer=None,
                proj_layers={"q_proj": block.attn.q,
                             "k_proj": block.attn.k,
                             "v_proj": block.attn.v},
                input_ids=None, position_ids=None)
            block.attn.pc_output.init_x(
                batch_size=B, seq_len=S, layer_type="linear_attn",
                device=device, layer=block.attn.output,
                proj_layers=None, input_ids=None, position_ids=None)
            block.mlp.pc_layer1.init_x(
                batch_size=B, seq_len=S, layer_type="fc1", device=device,
                layer=block.mlp.fc1,
                proj_layers=None, input_ids=None, position_ids=None)
            block.mlp.pc_layer2.init_x(
                batch_size=B, seq_len=S, layer_type="fc2", device=device,
                layer=block.mlp.fc2,
                proj_layers=None, input_ids=None, position_ids=None)

        self.output.pc_layer.init_x(
            batch_size=B, seq_len=S, layer_type="linear_output",
            device=device, layer=self.output.output,
            proj_layers=None, input_ids=None, position_ids=None)

        use_cuda, streams_or_futures = create_streams_or_futures(
            device, len(self.blocks) * 4 + 2)

        for t in range(self.config.T):
            td_mlp2 = (self.blocks[-1].mlp.pc_layer2.get_td_err("fc2")
                       if t > 0 else None)
            execute_parallel(
                use_cuda, streams_or_futures,
                self.output.pc_layer.forward,
                target_activity=target_logits, layer_type="linear_output",
                t=t, T=self.config.T, requires_update=True,
                td_err=td_mlp2, layer=self.output.output,
                layer_norm=None, proj_layers=None,
                input_ids=None, position_ids=None, flash=False)

            for idx in range(len(self.blocks) - 1, -1, -1):
                block = self.blocks[idx]
                next_target = (
                    self.blocks[idx + 1].attn.pc_qkv.get_x("attn")
                    if idx < len(self.blocks) - 1
                    else self.output.pc_layer.get_x("linear_output"))

                layer_norm2 = (block.ln2
                               if idx < len(self.blocks) - 1 else None)
                td_mlp1 = (block.mlp.pc_layer1.get_td_err("fc1")
                           if t > 0 else None)

                execute_parallel(
                    use_cuda, streams_or_futures,
                    block.mlp.pc_layer2.forward,
                    target_activity=next_target, layer_type="fc2",
                    t=t, T=self.config.T, requires_update=True,
                    td_err=td_mlp1, layer=block.mlp.fc2,
                    layer_norm=layer_norm2, proj_layers=None,
                    input_ids=None, position_ids=None, flash=False)

                td_attn_op = (block.attn.pc_output.get_td_err("linear_attn")
                              if t > 0 else None)

                execute_parallel(
                    use_cuda, streams_or_futures,
                    block.mlp.pc_layer1.forward,
                    target_activity=block.mlp.pc_layer2.get_x("fc2"),
                    layer_type="fc1",
                    t=t, T=self.config.T, requires_update=True,
                    td_err=td_attn_op, layer=block.mlp.fc1,
                    layer_norm=block.ln1, proj_layers=None,
                    input_ids=None, position_ids=None, flash=False)

                if idx == 0:
                    td_embed = (self.embedding.pc_layer.get_td_err("embed")
                                if t > 0 else None)
                else:
                    td_embed = (
                        self.blocks[idx - 1].mlp.pc_layer2.get_td_err("fc2")
                        if t > 0 else None)

                td_attn_qkv = (block.attn.pc_qkv.get_td_err("attn")
                               if t > 0 else None)

                execute_parallel(
                    use_cuda, streams_or_futures,
                    block.attn.pc_output.forward,
                    target_activity=block.mlp.pc_layer1.get_x("fc1"),
                    layer_type="linear_attn",
                    t=t, T=self.config.T, requires_update=True,
                    td_err=td_attn_qkv, layer=block.attn.output,
                    layer_norm=block.ln1, proj_layers=None,
                    input_ids=None, position_ids=None, flash=False)

                execute_parallel(
                    use_cuda, streams_or_futures,
                    block.attn.pc_qkv.forward,
                    target_activity=block.attn.pc_output.get_x("linear_attn"),
                    layer_type="attn",
                    t=t, T=self.config.T, requires_update=True,
                    td_err=td_embed, layer=None,
                    layer_norm=block.ln2,
                    proj_layers={"q_proj": block.attn.q,
                                 "k_proj": block.attn.k,
                                 "v_proj": block.attn.v},
                    input_ids=None, position_ids=None,
                    flash=getattr(self.config, "use_flash_attention", False),
                    use_cache=use_kv_cache,
                    kv_cache=(block.attn.kv_cache
                              if use_kv_cache else None))

                if use_kv_cache and t == self.config.T - 1:
                    block.attn.kv_cache = block.attn.pc_qkv._last_kv_cache

            execute_parallel(
                use_cuda, streams_or_futures,
                self.embedding.pc_layer.forward,
                target_activity=self.blocks[0].attn.pc_qkv.get_x("attn"),
                layer_type="embed",
                t=t, T=self.config.T, requires_update=True,
                td_err=None,
                layer={"word": self.embedding.word_embeddings,
                       "pos": self.embedding.position_embeddings},
                layer_norm=block.ln2, proj_layers=None,
                input_ids=input_ids, position_ids=position_ids, flash=False)

            synchronize_execution(use_cuda, streams_or_futures)

            # Apply wavelet-domain Hebbian updates (replaces step_linear's
            # spurious identity-matrix update with correct per-level updates).
            # Bidirectional: combine bu_err (from pc_layer1) with td_err
            # (from pc_layer2, the layer above) for richer gradient signal.
            for block in self.blocks:
                bu_err = block.mlp.pc_layer1._error_cache.get("fc1")
                if bu_err is not None:
                    td_err = block.mlp.pc_layer2._error_cache.get("fc2")
                    block.mlp.fc1.apply_hebbian_update(
                        bu_err, block.mlp.pc_layer1.local_lr,
                        td_err=td_err, td_alpha=self._td_alpha,
                        beta=self._beta)

        logits = self.output.pc_layer.get_mu("linear_output")
        return logits
