#!/usr/bin/env python3
"""
Gemma 4 E2B — LoRA Fine-tuning + FP Quantization Benchmark.

Three Groups (LoRA for fine-tuning + full-weight PTQ evaluation):

  Group A: LoRA fine-tune on all data → PTQ multi-precision
  Group B: Progressive LoRA bit-decomp on data tiers
  Group C: LoRA + SR on all data

Memory: ~10GB total (LoRA + BF16 model across GPUs).
"""

import os, math, json, argparse, time
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

# ═══════════════════════════════════════════════════════════
# Data
# ═══════════════════════════════════════════════════════════

class BinDataset(Dataset):
    def __init__(self, path, seq_len=1024):
        d = np.fromfile(path, dtype=np.uint32)
        self.data = torch.from_numpy(d.astype(np.int64))
        self.seq_len = seq_len
        self.n = max(0, (len(self.data)-1)//seq_len)
    def __len__(self): return self.n
    def __getitem__(self, i):
        s = i*self.seq_len; e = s+self.seq_len+1
        c = self.data[s:e]
        return {'input_ids':c[:-1],'labels':c[1:].clone(),'attention_mask':torch.ones(self.seq_len,dtype=torch.long)}

def mkloader(path, bs=2, sl=1024):
    return DataLoader(BinDataset(path,sl), batch_size=bs, shuffle=True, num_workers=0)

# ═══════════════════════════════════════════════════════════
# LoRA setup
# ═══════════════════════════════════════════════════════════

def add_lora(model, r=8):
    from peft import get_peft_model, LoraConfig, TaskType
    # Gemma 4 uses custom linear layers — target by name pattern
    target = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    config = LoraConfig(r=r, lora_alpha=16, target_modules=target,
                        lora_dropout=0.0, bias="none", task_type=TaskType.CAUSAL_LM)
    try:
        model = get_peft_model(model, config)
    except ValueError:
        # Fallback: find actual linear module names
        linear_names = [n for n,m in model.named_modules()
                        if isinstance(m, nn.Linear) and any(t in n for t in target)]
        config = LoraConfig(r=r, lora_alpha=16, target_modules=list(set(
            n.split('.')[-1] for n in linear_names)),
            lora_dropout=0.0, bias="none", task_type=TaskType.CAUSAL_LM)
        model = get_peft_model(model, config)
    model.enable_input_require_grads()
    return model

# ═══════════════════════════════════════════════════════════
# Multi-precision PTQ eval
# ═══════════════════════════════════════════════════════════

class FPQuant:
    def __init__(self, bits, per_channel=True):
        self.bits = bits; self.per_channel = per_channel
    def quantize(self, x, sr=False):
        n = (1<<(self.bits-1))-1
        if self.per_channel and x.dim()>=2:
            am = x.abs().max(dim=-1,keepdim=True)[0].clamp(min=1e-12)
        else:
            am = x.abs().max().clamp(min=1e-12)
        xn = x/(am+1e-12)*n
        xq = (xn+torch.rand_like(xn)-0.5).round() if sr else xn.round()
        return xq.clamp(-n,n)/n*am

@torch.no_grad()
def eval_ppl(model, loader, mb=30):
    model.eval(); tl,nn=0.0,0
    for i,b in enumerate(loader):
        if i>=mb: break
        out=model(input_ids=b['input_ids'].to(model.device),labels=b['labels'].to(model.device))
        tl+=out.loss.item()*b['input_ids'].numel(); nn+=b['input_ids'].numel()
    return math.exp(tl/max(nn,1))

def ptq_multiprecision(model, loader, bits=[8,4,2,1]):
    """Quantize full pretrained weights (not LoRA) and eval."""
    r={}
    for b in bits:
        q=FPQuant(b)
        saved={}
        for n,p in model.named_parameters():
            if 'lora' not in n and p.dim()>=2:
                saved[n]=p.data.clone()
                # Quantize on CPU to avoid GPU OOM
                pcpu=p.data.cpu()
                p.data=q.quantize(pcpu).to(p.device)
        ppl=eval_ppl(model,loader)
        for n,p in model.named_parameters():
            if n in saved: p.data=saved[n]
        r[b]=round(ppl,2)
        print(f"  PTQ-{b}bit: PPL={ppl:.2f}")
    return r

# ═══════════════════════════════════════════════════════════
# Group A: LoRA + all data → PTQ
# ═══════════════════════════════════════════════════════════

def group_a(model, tokenizer, data_dir, out_dir, steps=500):
    print("\n"+"="*50+"\nGROUP A: LoRA FT + PTQ\n"+"="*50)
    model = add_lora(model)
    model.train()
    model.gradient_checkpointing_enable()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)

    loaders = {f.replace('.bin',''): mkloader(os.path.join(data_dir,f),bs=1,sl=512)
               for f in sorted(os.listdir(data_dir)) if f.endswith('.bin')}

    gs=0
    for ep in range(2):
        for nm,ld in loaders.items():
            for b in ld:
                if gs>=steps: break
                out=model(input_ids=b['input_ids'].to(model.device),labels=b['labels'].to(model.device))
                opt.zero_grad(); out.loss.backward(); opt.step()
                gs+=1
                if gs%100==0: print(f"  step {gs}: loss={out.loss.item():.3f}")
            if gs>=steps: break

    os.makedirs(out_dir,exist_ok=True)
    model.save_pretrained(out_dir)
    print(f"Saved to {out_dir}")

    # PTQ eval
    print("\nPTQ multi-precision:")
    r = ptq_multiprecision(model, list(loaders.values())[0])
    with open(os.path.join(out_dir,'ptq.json'),'w') as f: json.dump(r,f)
    return r

# ═══════════════════════════════════════════════════════════
# Group B: Progressive LoRA + data tiers
# ═══════════════════════════════════════════════════════════

def group_b(model, tokenizer, data_dir, out_dir):
    print("\n"+"="*50+"\nGROUP B: Progressive LoRA + Data Tiers\n"+"="*50)
    model = add_lora(model)
    model.train()
    model.gradient_checkpointing_enable()

    tiers = sorted([f for f in os.listdir(data_dir) if f.endswith('.bin')],
                   key=lambda x: os.path.getsize(os.path.join(data_dir,x)), reverse=True)
    results = []

    for phase, tier_file in enumerate(tiers):
        tier_name = tier_file.replace('.bin','')
        loader = mkloader(os.path.join(data_dir,tier_file), bs=1, sl=512)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-4*(0.8**phase))

        t0=time.time()
        for step,b in enumerate(loader):
            if step>=150: break
            out=model(input_ids=b['input_ids'].to(model.device),labels=b['labels'].to(model.device))
            opt.zero_grad(); out.loss.backward(); opt.step()
        elapsed=time.time()-t0
        ppl=eval_ppl(model,loader)
        print(f"  Tier {phase} ({tier_name}): PPL={ppl:.2f}, {elapsed:.0f}s")
        results.append({'tier':phase,'name':tier_name,'ppl':ppl})

    # Multi-precision
    print("\nPTQ multi-precision:")
    r = ptq_multiprecision(model, mkloader(os.path.join(data_dir,tiers[0]),bs=1,sl=512))
    for k,v in r.items(): results[-1][f'ptq_{k}bit']=v

    os.makedirs(out_dir,exist_ok=True)
    model.save_pretrained(out_dir)
    with open(os.path.join(out_dir,'results.json'),'w') as f: json.dump(results,f)
    return results

# ═══════════════════════════════════════════════════════════
# Group C: LoRA + SR on all data
# ═══════════════════════════════════════════════════════════

def group_c(model, tokenizer, data_dir, out_dir, steps=500):
    print("\n"+"="*50+"\nGROUP C: LoRA + Stochastic Rounding\n"+"="*50)
    model = add_lora(model)
    model.train()
    model.gradient_checkpointing_enable()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)

    all_bins = [f for f in os.listdir(data_dir) if f.endswith('.bin')]
    loader = mkloader(os.path.join(data_dir,all_bins[0]), bs=1, sl=512)
    q = FPQuant(4)  # FP4 with SR

    for step,b in enumerate(loader):
        if step>=steps: break
        # Apply FP4+SR quantization to full weights
        saved={}
        for n,p in model.named_parameters():
            if 'lora' not in n and p.dim()>=2:
                saved[n]=p.data.clone()
                pcpu=p.data.cpu()
                p.data=q.quantize(pcpu, sr=True).to(p.device)
        out=model(input_ids=b['input_ids'].to(model.device),labels=b['labels'].to(model.device))
        opt.zero_grad(); out.loss.backward()
        for n,p in model.named_parameters():
            if n in saved: p.data=saved[n]
        opt.step()
        if step%100==0: print(f"  step {step}: loss={out.loss.item():.3f}")

    os.makedirs(out_dir,exist_ok=True)
    model.save_pretrained(out_dir)
    r = ptq_multiprecision(model,loader)
    with open(os.path.join(out_dir,'ptq.json'),'w') as f: json.dump(r,f)
    return r

# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--group',required=True,choices=['A','B','C'])
    p.add_argument('--model_dir',default='models/gemma4-e2b')
    p.add_argument('--data_dir',default='data/gemma4_tiers')
    p.add_argument('--output_dir',default=None)
    p.add_argument('--steps',type=int,default=500)
    a=p.parse_args()
    a.output_dir=a.output_dir or f'checkpoints/gemma4_group{a.group}'

    print("Loading Gemma 4 E2B...")
    m=AutoModelForCausalLM.from_pretrained(a.model_dir,torch_dtype=torch.bfloat16,
        device_map='auto',max_memory={i:'18GiB' for i in range(8)},
        trust_remote_code=True,local_files_only=True)
    tok=AutoTokenizer.from_pretrained(a.model_dir,trust_remote_code=True,local_files_only=True)
    if tok.pad_token is None: tok.pad_token=tok.eos_token

    # Pre-finetune PTQ baseline
    print("\nPre-finetune PTQ baseline:")
    loader=mkloader(os.path.join(a.data_dir,sorted(os.listdir(a.data_dir))[0]),bs=1,sl=512)
    baseline=ptq_multiprecision(m,loader)

    if a.group=='A':
        group_a(m,tok,a.data_dir,a.output_dir,a.steps)
    elif a.group=='B':
        group_b(m,tok,a.data_dir,a.output_dir)
    else:
        group_c(m,tok,a.data_dir,a.output_dir,a.steps)

if __name__=='__main__':
    main()
