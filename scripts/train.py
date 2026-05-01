#!/usr/bin/env python3
"""Train WaveletPCTransformer (QuantiMORK) or baseline PCTransformer on Tiny Shakespeare.

Usage:
    python scripts/train.py --mode wavelet   # QuantiMORK-THRML (default)
    python scripts/train.py --mode baseline  # PC-Transformers baseline
"""

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

# Add vendor to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
VENDOR_DIR = os.path.join(PROJECT_ROOT, "vendor", "PC-Transformers")
sys.path.insert(0, VENDOR_DIR)

from predictive_coding.config import GPTConfig
from predictive_coding.pc_layer import PCLayer
from data_preparation.dataloader import get_loaders
from data_preparation.config import vocab_size


def set_seed(seed):
    import random
    random.seed(seed)
    torch.manual_seed(seed)


def evaluate(model, config, dataloader, max_batches=None, device=None):
    """Lightweight evaluate — no DDP, no bert_score."""
    model.eval()
    total_ce = 0.0
    total_energy = 0.0
    count = 0

    for batch_idx, batch in enumerate(dataloader):
        if max_batches and batch_idx >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        targets = batch["target_ids"].to(device)
        if targets.max() >= vocab_size:
            targets = torch.clamp(targets, max=vocab_size - 1)

        logits = model(targets, input_ids)
        ce = F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=0)
        total_ce += ce.item()

        energies = []
        for m in model.modules():
            if isinstance(m, PCLayer) and hasattr(m, "get_energy"):
                e = m.get_energy()
                if e is not None and not (isinstance(e, float) and math.isnan(e)):
                    energies.append(e)
        total_energy += (sum(energies) / len(energies)) if energies else ce.item()
        count += 1

    avg_ce = total_ce / max(count, 1)
    avg_e = total_energy / max(count, 1)
    avg_ppl = math.exp(avg_ce) if avg_ce < 100 else float("inf")
    return avg_e, avg_ppl


def make_config(mode="wavelet", num_epochs=5):
    """Create training config — smaller than PC-Transformers' default for fast iteration."""
    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=64,
        lr=0.001,
        peak_learning_rate=0.001,
        warmup_steps=20,
        n_embed=128,          # smaller for CPU training
        dropout=0.1,
        T=2,
        num_heads=4,
        n_blocks=4,
        batch_size=8,
        num_epochs=num_epochs,
        update_bias=False,
        internal_energy_fn_name="pc_e",
        output_energy_fn_name="pc_e",
        combined_internal_weight=0.8,
        combined_output_weight=0.2,
        use_flash_attention=False,
        alpha=0.5,
    )
    if mode == "wavelet":
        config.wavelet_n_levels = 3
        config.td_alpha = 0.5
        config.beta = 0.1
    config.attn_mode = "attn"
    return config


def train_epoch_fluid(model, dataloader, config, global_step, device,
                     optimizer, max_steps=None):
    """Training step for attn_mode='fluid' — standard backprop with Adam.

    The fluid block runs internal energy descent inside its forward, so we
    don't need the vendor PC iterative scheme. Weights update via
    `loss.backward()` + `optimizer.step()`.
    """
    model.train()
    total_ce_loss = 0.0
    batch_count = 0
    for batch_idx, batch in enumerate(dataloader):
        if max_steps is not None and global_step >= max_steps:
            break
        input_ids = batch["input_ids"].to(device)[:, :config.block_size]
        target_ids = batch["target_ids"].to(device)[:, :config.block_size]
        if target_ids.max() >= vocab_size:
            target_ids = torch.clamp(target_ids, max=vocab_size - 1)
        optimizer.zero_grad()
        logits = model(target_ids, input_ids)
        ce_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_ids.reshape(-1),
            ignore_index=0,
        )
        # §10.7 three-part free energy: task loss + reaction PE + value
        # anchor (regional v_field <-> converged rho consistency).
        react_total = 0.0
        anchor_total = 0.0
        for blk in model.blocks:
            fe = getattr(blk.fluid_pc, "last_free_energy", {})
            if fe:
                react_total = react_total + fe["reaction"]
                anchor_total = anchor_total + fe["v_anchor"]
        w_react = getattr(config, "fluid_react_weight", 0.01)
        w_anchor = getattr(config, "fluid_anchor_weight", 0.1)
        loss = ce_loss + w_react * react_total + w_anchor * anchor_total
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_ce_loss += ce_loss.item()
        batch_count += 1
        global_step += 1
        ppl = math.exp(ce_loss.item()) if ce_loss.item() < 100 else float("inf")
        rval = react_total.item() if hasattr(react_total, "item") else 0.0
        aval = anchor_total.item() if hasattr(anchor_total, "item") else 0.0
        print(f"  step {global_step} | CE={ce_loss.item():.4f} "
              f"PPL={ppl:.1f} react={rval:.4f} anchor={aval:.4f}")
    avg_ce = total_ce_loss / max(batch_count, 1)
    avg_ppl = math.exp(avg_ce) if avg_ce < 100 else float("inf")
    return avg_ce, avg_ppl, global_step


def build_model(config, mode):
    if mode == "wavelet":
        from quantimork_thrml.model import WaveletPCTransformer
        model = WaveletPCTransformer(config)
    else:
        from model_architecture.pc_t_model import PCTransformer
        model = PCTransformer(config)
    return model


def train_epoch(model, dataloader, config, global_step, device):
    model.train()
    total_ce_loss = 0.0
    total_energy = 0.0
    batch_count = 0

    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)

        if target_ids.max() >= vocab_size:
            target_ids = torch.clamp(target_ids, max=vocab_size - 1)

        # Learning rate schedule
        if global_step < config.warmup_steps:
            lr = config.lr + global_step / config.warmup_steps * (
                config.peak_learning_rate - config.lr)
        else:
            lr = config.peak_learning_rate

        for module in model.modules():
            if hasattr(module, "local_lr"):
                module.set_learning_rate(lr)

        global_step += 1

        logits = model(target_ids, input_ids)
        ce_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_ids.view(-1),
            ignore_index=0,
        )
        total_ce_loss += ce_loss.item()

        # Collect PC energies
        internal_energies = []
        for module in model.modules():
            if isinstance(module, PCLayer) and hasattr(module, "get_energy"):
                energy = module.get_energy()
                if energy is not None and not (
                    isinstance(energy, float) and math.isnan(energy)):
                    internal_energies.append(energy)

        avg_energy = (sum(internal_energies) / len(internal_energies)
                      if internal_energies else ce_loss.item())
        total_energy += avg_energy
        batch_count += 1

        # KL energy component (weight decay penalty)
        kl_energy = 0.0
        beta = getattr(config, "beta", 0.0)
        if beta > 0 and hasattr(model, "_beta"):
            for block in model.blocks:
                wl = block.mlp.wavelet
                for dt in wl.detail_transforms:
                    kl_energy += beta * 0.5 * float(dt.weight.data.pow(2).sum())
                kl_energy += beta * 0.5 * float(
                    wl.approx_transform.weight.data.pow(2).sum())

        if (batch_idx + 1) % 20 == 0:
            ppl = math.exp(ce_loss.item()) if ce_loss.item() < 100 else float("inf")
            kl_str = f" KL={kl_energy:.4f}" if kl_energy > 0 else ""
            print(f"  batch {batch_idx+1}/{len(dataloader)} | "
                  f"CE={ce_loss.item():.4f} PPL={ppl:.1f} "
                  f"energy={avg_energy:.4f}{kl_str}")

    avg_ce = total_ce_loss / max(batch_count, 1)
    avg_e = total_energy / max(batch_count, 1)
    avg_ppl = math.exp(avg_ce) if avg_ce < 100 else float("inf")
    return avg_e, avg_ppl, global_step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["wavelet", "baseline"],
                        default="wavelet")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--td-alpha", type=float, default=0.5,
                        help="Top-down error weight for bidirectional Hebbian update")
    parser.add_argument("--beta", type=float, default=0.1,
                        help="KL divergence regularization strength")
    parser.add_argument("--attn-mode", choices=["attn", "fluid"],
                        default="attn",
                        help="'fluid' replaces attention with FluidPCBlock (IFN §10.6)")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Cap total training steps (smoke test)")
    parser.add_argument("--fluid-outer-iters", type=int, default=2)
    parser.add_argument("--mpc-horizon", type=int, default=2)
    parser.add_argument("--mpc-inner-steps", type=int, default=3)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cpu")

    config = make_config(args.mode, args.epochs)
    if args.mode == "wavelet":
        config.td_alpha = args.td_alpha
        config.beta = args.beta
        config.attn_mode = args.attn_mode
        if args.attn_mode == "fluid":
            config.fluid_outer_iters = args.fluid_outer_iters
            config.mpc_horizon = args.mpc_horizon
            config.mpc_inner_steps = args.mpc_inner_steps
    model = build_model(config, args.mode).to(device)
    fluid_mode = (args.mode == "wavelet" and args.attn_mode == "fluid")
    optimizer = (torch.optim.Adam(model.parameters(), lr=config.lr)
                 if fluid_mode else None)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"=== Training {args.mode.upper()} model ===")
    print(f"Parameters: {n_params/1e6:.2f}M")
    print(f"Config: n_embed={config.n_embed} n_blocks={config.n_blocks} "
          f"T={config.T} epochs={config.num_epochs}")

    train_loader, valid_loader, test_loader = get_loaders(distributed=False)

    global_step = 0
    results = {"mode": args.mode, "epochs": [], "config": {
        "n_embed": config.n_embed, "n_blocks": config.n_blocks,
        "T": config.T, "num_heads": config.num_heads,
        "n_params": n_params,
    }}

    start = time.time()
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch+1}/{config.num_epochs}")
        if fluid_mode:
            train_energy, train_ppl, global_step = train_epoch_fluid(
                model, train_loader, config, global_step, device,
                optimizer, max_steps=args.max_steps)
            if args.max_steps is not None and global_step >= args.max_steps:
                break
        else:
            train_energy, train_ppl, global_step = train_epoch(
                model, train_loader, config, global_step, device)

        model.eval()
        with torch.no_grad():
            val_energy, val_ppl = evaluate(
                model, config, valid_loader, max_batches=None, device=device)

        print(f"  train PPL={train_ppl:.1f} energy={train_energy:.4f}")
        print(f"  val   PPL={val_ppl:.1f} energy={val_energy:.4f}")

        results["epochs"].append({
            "epoch": epoch + 1,
            "train_ppl": train_ppl,
            "train_energy": train_energy,
            "val_ppl": val_ppl,
            "val_energy": val_energy,
        })

    elapsed = time.time() - start
    print(f"\nTraining completed in {elapsed:.1f}s")

    # Save results
    os.makedirs("results", exist_ok=True)
    out_path = f"results/{args.mode}_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")

    # Save model
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = f"checkpoints/{args.mode}_final.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Model saved to {ckpt_path}")


if __name__ == "__main__":
    main()
