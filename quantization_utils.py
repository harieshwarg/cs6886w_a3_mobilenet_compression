
import torch, torch.nn as nn
import numpy as np
from collections import OrderedDict

def qparams_symmetric(t, num_bits):
    qmax = (1<<(num_bits-1)) - 1
    max_abs = t.detach().abs().max().item()
    scale = 1.0 if max_abs==0 else max_abs/qmax
    return scale, 0, qmax

def quantize_per_tensor_sym(t, num_bits):
    scale, zp, qmax = qparams_symmetric(t, num_bits)
    q = torch.clamp(torch.round(t/scale), -qmax-1, qmax).to(torch.int8)
    return q, scale, zp

def pack_weights_state_dict(model, wbits):
    sd=model.state_dict()
    qsd=OrderedDict(); meta={"wbits":wbits,"items":[]}
    for name,p in sd.items():
        if not p.dtype.is_floating_point:
            qsd[name] = p.cpu().numpy()
            meta["items"].append({"name":name,"kind":"nonfloat","shape":tuple(p.shape),"dtype":str(p.dtype)})
            continue
        q,s,zp = quantize_per_tensor_sym(p.to(torch.float32), wbits)
        qsd[name]=q.cpu().numpy()
        meta["items"].append({"name":name,"kind":"float","scale":float(s),"zp":int(zp),"shape":tuple(p.shape)})
    return qsd, meta

def save_quant_weights(model, out_dir, tag, wbits):
    import os, json
    os.makedirs(out_dir, exist_ok=True)
    qsd, meta = pack_weights_state_dict(model, wbits)
    npz_path = f"{out_dir}/mnv2_w{wbits}_{tag}.npz"
    jsn_path = f"{out_dir}/mnv2_w{wbits}_{tag}.json"
    np.savez_compressed(npz_path, **qsd)
    with open(jsn_path,"w") as f: json.dump(meta,f)
    return npz_path, jsn_path, meta

def estimate_weight_bytes_incl_overheads(model, wbits, meta_items):
    payload=0
    for name,p in model.state_dict().items():
        if p.dtype.is_floating_point:
            payload += (p.numel()*wbits)/8.0
        else:
            payload += p.numel()*p.element_size()
    n_scales = sum(1 for it in meta_items if it.get("kind")=="float")
    overhead = n_scales*4 + 1024
    return int(payload+overhead)

class ActCalibrator:
    def __init__(self): self.stats={}
    def _hook(self,name):
        def f(mod,inp,out):
            m = float(out.detach().abs().max().item())
            self.stats[name] = max(m, self.stats.get(name,0.0))
        return f
    def attach(self,model):
        self.handles=[]
        for name,m in model.named_modules():
            if isinstance(m,(nn.Conv2d,nn.Linear)):
                self.handles.append(m.register_forward_hook(self._hook(name)))
    def detach(self):
        for h in self.handles: h.remove()

def calibrate_activations(model, loader, max_batches, device):
    model.eval()
    calib=ActCalibrator(); calib.attach(model)
    with torch.no_grad():
        for i,(x,_) in enumerate(loader):
            x=x.to(device)
            _=model(x)
            if i>=max_batches-1: break
    calib.detach()
    return calib.stats

def fake_quant_activation(x, max_abs, abits):
    qmax=(1<<(abits-1))-1
    scale = 1.0 if max_abs==0 else max_abs/qmax
    q = torch.clamp(torch.round(x/scale), -qmax-1, qmax)
    return q*scale

class OutputFakeQuant:
    def __init__(self,model,table,abits):
        self.model=model; self.table=table; self.abits=abits; self.handles=[]
    def _hook(self,name):
        def h(mod,inp,out):
            max_abs = self.table.get(name, None)
            if max_abs is None:
                for k,v in self.table.items():
                    if k.endswith(name): max_abs=v; break
            if max_abs is None:
                max_abs=float(out.detach().abs().max())
            return fake_quant_activation(out,max_abs,self.abits)
        return h
    def attach(self):
        for name,m in self.model.named_modules():
            if isinstance(m,(nn.Conv2d,nn.Linear)):
                self.handles.append(m.register_forward_hook(self._hook(name)))
    def detach(self):
        for h in self.handles: h.remove()

def evaluate_with_output_fq(model, loader, act_max_table, abits, device):
    from train_baseline import evaluate
    model.eval()
    fq = OutputFakeQuant(model, act_max_table, abits)
    try:
        fq.attach()
        loss, acc = evaluate(model, loader)
    finally:
        fq.detach()
    return loss, acc
