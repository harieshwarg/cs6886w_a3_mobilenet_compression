
import torch
from seed_utils import DEVICE
from quantization_utils import (
    save_quant_weights, estimate_weight_bytes_incl_overheads,
    calibrate_activations, evaluate_with_output_fq
)
from train_baseline import train_baseline

def run_q2(wbits=8, abits=8, calib_batches=8):
    model,_ = train_baseline(epochs=1)  # <- for full run set epochs=200 externally
    fp32_bytes = sum(p.numel()*4 for p in model.state_dict().values()
                     if p.dtype.is_floating_point)

    npz_path, jsn_path, meta = save_quant_weights(model, "checkpoints", "q2", wbits)
    est_bytes = estimate_weight_bytes_incl_overheads(model, wbits, meta["items"])

    print("FP32 weights:", fp32_bytes)
    print("Quantized est bytes:", est_bytes)
    print("Compression ratio:", fp32_bytes/est_bytes)

    act_table = calibrate_activations(model, loader=None, max_batches=calib_batches, device=DEVICE)
    _, acc_c = evaluate_with_output_fq(model, None, act_table, abits, DEVICE)
    print("Compressed eval acc:", acc_c)
