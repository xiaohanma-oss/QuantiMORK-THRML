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
    return config


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

        if (batch_idx + 1) % 20 == 0:
            ppl = math.exp(ce_loss.item()) if ce_loss.item() < 100 else float("inf")
            print(f"  batch {batch_idx+1}/{len(dataloader)} | "
                  f"CE={ce_loss.item():.4f} PPL={ppl:.1f} energy={avg_energy:.4f}")

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
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cpu")

    config = make_config(args.mode, args.epochs)
    model = build_model(config, args.mode).to(device)

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
