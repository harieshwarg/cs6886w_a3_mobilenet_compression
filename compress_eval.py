# compress_eval.py
#
# Q2/Q3/Q4: compression experiments for MobileNet-V2 on CIFAR-10
# - Q2: single (wbits, abits) config
# - Q3: sweep over several (wbits, abits) pairs → CSV table
# - Q4: compression ratio + model size analysis

import argparse
import pandas as pd
import torch

from seed_utils import DEVICE, set_seed
from data_cifar10 import get_cifar10_loaders
from model_mobilenetv2 import mobilenet_v2_cifar10
from quantization_utils import (
    save_quant_weights,
    estimate_weight_bytes_incl_overheads,
    calibrate_activations,
    evaluate_with_output_fq,
)

# ---------- helpers ----------

def load_baseline(ckpt_path: str):
    """Load FP32 MobileNet-V2 baseline from checkpoint."""
    model = mobilenet_v2_cifar10()
    state = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    model.to(DEVICE)
    return model


def run_q2_single(model, valloader, testloader, wbits: int, abits: int, tag: str):
    """Single compression config: Q2 (and part of Q4)."""
    print(f"[Q2] Using WBITS={wbits}, ABITS={abits} on model: {type(model).__name__}")
    print("     Layers compressed: Conv2d & Linear (BN/Dropout/etc. left FP32).")

    # 1) Weight quantization + size estimate
    npz_path, jsn_path, meta = save_quant_weights(
        model, out_dir="checkpoints", tag=tag, wbits=wbits
    )
    fp32_weight_bytes = sum(
        p.numel() * 4 for p in model.parameters() if p.dtype.is_floating_point
    )
    est_bytes = estimate_weight_bytes_incl_overheads(model, wbits, meta["items"])
    ratio = fp32_weight_bytes / max(1, est_bytes)

    # 2) Activation calibration (few val batches)
    act_table = calibrate_activations(model, valloader, max_batches=8, device=DEVICE)

    # 3) Simulated inference with fake-quantized outputs
    _, test_acc_c = evaluate_with_output_fq(
        model, testloader, act_table, abits, device=DEVICE
    )

    print("\n=== Q2 Results ===")
    print(f"Simulated compressed Test Acc   : {test_acc_c:.2f}%  (ABITS={abits})")
    print(f"Estimated weight storage (bytes): {est_bytes}")
    print(f"FP32 weight size (bytes)        : {fp32_weight_bytes}")
    print(f"Compression ratio (weights)     : {ratio:.2f}x")
    print(f"Quantized archive saved         : {npz_path}")
    print(f"Metadata JSON                   : {jsn_path}\n")

    return {
        "fp32_weight_bytes": fp32_weight_bytes,
        "est_bytes": est_bytes,
        "ratio": ratio,
        "test_acc": float(test_acc_c),
        "act_table": act_table,
    }


def run_q3_sweep(model, testloader, act_table, fp32_weight_bytes, out_csv: str):
    """Sweep several (wbits, abits) configs and write CSV for parallel-coords plot."""
    sweep_confs = [
        {"wbits": 8, "abits": 8},
        {"wbits": 8, "abits": 6},
        {"wbits": 8, "abits": 4},
        {"wbits": 6, "abits": 8},
        {"wbits": 6, "abits": 6},
        {"wbits": 6, "abits": 4},
        {"wbits": 4, "abits": 8},
        {"wbits": 4, "abits": 6},
    ]

    rows = []
    print("\n[Q3] Starting compression sweep...\n")
    for i, cfg in enumerate(sweep_confs, start=1):
        wbits, abits = cfg["wbits"], cfg["abits"]
        tag = f"q3_w{wbits}_a{abits}"
        print(f"Run {i}/{len(sweep_confs)} → weight_quant_bits={wbits}, activation_quant_bits={abits}")

        # quantize weights & estimate size
        _, _, meta = save_quant_weights(model, out_dir="checkpoints", tag=tag, wbits=wbits)
        est_bytes = estimate_weight_bytes_incl_overheads(model, wbits, meta["items"])
        ratio = fp32_weight_bytes / max(1, est_bytes)
        size_mb = est_bytes / (1024 * 1024)

        # evaluate accuracy with fake-quant activations
        _, test_acc_c = evaluate_with_output_fq(
            model, testloader, act_table, abits, device=DEVICE
        )
        print(f"   → quantized_acc={test_acc_c:.2f}% | compression_ratio={ratio:.2f}x | model_size_mb={size_mb:.3f}")

        rows.append(
            dict(
                activation_quant_bits=abits,
                weight_quant_bits=wbits,
                compression_ratio=float(ratio),
                model_size_mb=float(size_mb),
                quantized_acc=float(test_acc_c),
            )
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print("\n[Q3] Sweep complete. Saved table →", out_csv)
    print(df)
    return df


def run_q4_analysis(fp32_weight_bytes, est_bytes, wbits: int, abits: int):
    """Print Q4(a–d) analysis for a given (wbits, abits) config."""
    print("\n=== Q4 Results ===")

    # (a) model compression ratio (here we use weights as dominant)
    cr_model = fp32_weight_bytes / est_bytes
    print(f"(a) Model compression ratio         : {cr_model:.3f}x")

    # (b) compression ratio of weights
    cr_weights_theoretical = 32.0 / wbits
    print(f"(b) Weight-only compression ratio   : {cr_weights_theoretical:.3f}x (theoretical)")
    print(f"    Measured (with overheads)       : {fp32_weight_bytes/est_bytes:.3f}x")

    # (c) compression ratio of activations
    cr_acts = 32.0 / abits
    print(f"(c) Activation compression ratio    : {cr_acts:.3f}x")
    print(f"    (Defined by activation bit-width ABITS={abits})")

    # (d) final compressed model size
    size_mb = est_bytes / (1024 * 1024)
    print(f"(d) Final compressed model size (MB): {size_mb:.3f} MB\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="checkpoints/baseline_fp32.pt",
                        help="Path to FP32 baseline checkpoint (from train_baseline.py)")
    parser.add_argument("--mode", choices=["q2", "q3"], required=True,
                        help="q2 = single config; q3 = sweep")
    parser.add_argument("--wbits", type=int, default=8,
                        help="Weight bit-width for Q2 (and reference for Q3)")
    parser.add_argument("--abits", type=int, default=8,
                        help="Activation bit-width for Q2 (and reference for Q3)")
    parser.add_argument("--out_csv", type=str, default="q3_compression_sweep.csv",
                        help="Output CSV path for Q3 sweep table")
    args = parser.parse_args()

    set_seed(42)
    trainloader, valloader, testloader = get_cifar10_loaders(batch_size=128)

    # Load trained FP32 baseline
    model = load_baseline(args.ckpt)

    if args.mode == "q2":
        stats = run_q2_single(
            model,
            valloader,
            testloader,
            wbits=args.wbits,
            abits=args.abits,
            tag="q2",
        )
        run_q4_analysis(
            fp32_weight_bytes=stats["fp32_weight_bytes"],
            est_bytes=stats["est_bytes"],
            wbits=args.wbits,
            abits=args.abits,
        )
    else:
        # For Q3 we first run Q2 once to get act_table + fp32 bytes
        stats = run_q2_single(
            model,
            valloader,
            testloader,
            wbits=args.wbits,
            abits=args.abits,
            tag="q3_ref",
        )
        _ = run_q3_sweep(
            model,
            testloader,
            act_table=stats["act_table"],
            fp32_weight_bytes=stats["fp32_weight_bytes"],
            out_csv=args.out_csv,
        )
        print("\n[Q3] Use the CSV to plot a parallel-coordinates chart for reporting.")

if __name__ == "__main__":
    main()
