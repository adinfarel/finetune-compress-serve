import json
import math
import time
import torch
import numpy as np
from pathlib import Path
from typing import Optional
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    BitsAndBytesConfig,
    FastVlmPreTrainedModel,
    DataCollatorForSeq2Seq,
)
from datasets import load_dataset

from fcs.compress.config import CompressConfig, EvalConfig

@torch.no_grad()
def compute_perplexity(
    model,
    tokenizer,
    n_samples: int = 500,
    max_seq_length: int = 512,
    batch_size: int = 8,
    seed: int = 42,
    device: Optional[str] = None
) -> dict:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    from datasets import load_dataset
    raw = load_dataset("tatsu-lab/alpaca", split="train")
    raw = raw.train_test_split(test_size=0.05, seed=seed)["test"]
    raw = raw.select(range(min(n_samples, len(raw))))
    
    def tokenize(ex):
        text = (
            f"### Instruction:\n{ex['instruction']}\n\n"
            f"### Response:\n{ex['output']}"
        )
        tokens = tokenizer(
            text, truncation=True,
            max_length=max_seq_length, padding=False,
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens 

    tokenized = raw.map(tokenize, remove_columns=raw.column_names, batched=False)
    
    # collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest", return_tensors="pt")
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,        
        padding=True,       
        return_tensors="pt"
    )
    loader = DataLoader(tokenized, batch_size=batch_size, collate_fn=collator, shuffle=False) #type: ignore
    
    model.eval()
    total_loss, total_tokens = 0.0, 0
    
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        labels[labels == tokenizer.pad_token_id] = -100
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        n_tokens = (labels != -100).sum().item()
        total_loss += outputs.loss.item() * n_tokens
        total_tokens += n_tokens
    
    avg_loss = total_loss / total_tokens
    return {
        "perplexity": round(math.exp(avg_loss), 4),
        "avg_loss": round(avg_loss, 6),
        "n_tokens": total_tokens,
    }

import re

@torch.no_grad()
def compute_gsm8k(
    model,
    tokenizer,
    n_samples: int = 24,
    max_new_tokens: int = 256,
    batch_size: int = 8,
    device: Optional[str] = None,
) -> dict:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
      
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    ds = list(load_dataset("openai/gsm8k", "main", split=f"test", streaming=True).take(n_samples))
    model.eval()
    correct = 0
    
    for i in range(0, len(ds), batch_size):
        batch_exs = ds[i : i + batch_size]
        
        prompts = [
            f"Below is a math problem. Solve it step by step.\n\n### Problem:\n{ex['question']}\n\n### Solution:\n"
            for ex in batch_exs
        ]
        
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True, 
        )
        
        input_len = inputs["input_ids"].shape[1]
        for idx, ex in enumerate(batch_exs):
            generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

            nums_pred = re.findall(r"-?\d+(?:\.\d+)?", generated.replace(",", ""))
            nums_gold = re.findall(r"-?\d+(?:\.\d+)?", ex["answer"].replace(",", ""))

            pred = float(nums_pred[-1]) if nums_pred else None
            gold = float(nums_gold[-1]) if nums_gold else None

            if pred is not None and gold is not None and abs(pred - gold) < 1e-6:
                correct += 1
    
    return {
        "gsm8k_accuracy": round(correct / n_samples, 4),
        "gsm8k_correct": correct,
        "gsm8k_total": n_samples,
    }

@torch.no_grad()
def measure_latency(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    warmup_runs: int = 3,
    benchmark_runs: int = 20,
    device: Optional[str] = None,
) -> dict:
    """Measure TTFT (Prefill) and ITL (Decode)"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    for _ in range(warmup_runs):
        _ = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    ttfts, itls = [], []
    
    for _ in range(benchmark_runs):
        torch.cuda.synchronize()
        
        t_start = time.perf_counter()
        
        first_token = model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        torch.cuda.synchronize()
        ttft = time.perf_counter() - t_start
        ttfts.append(ttft * 1000) # ms
        
        torch.cuda.synchronize()
        t_dcd_start = time.perf_counter()
        full_output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        
        torch.cuda.synchronize()
        t_dcd_total = time.perf_counter() - t_dcd_start
        
        n_generated = full_output.shape[1] - inputs["input_ids"].shape[1]
        if n_generated > 1:
            t_itl_total = t_dcd_total - ttft
            itl = (t_itl_total / (n_generated - 1)) * 1000
            itls.append(itl)
        
    ttfts_arr = np.array(ttfts)
    itls_arr = np.array(itls)
    
    return {
        "ttft_p50_ms": round(float(np.percentile(ttfts_arr, 50)), 3),
        "ttft_p99_ms": round(float(np.percentile(ttfts_arr, 99)), 3),
        "itl_p50_ms": round(float(np.percentile(itls_arr, 50)), 3),
        "itl_p99_ms": round(float(np.percentile(itls_arr, 99)), 3),
        "throughput_per_sec": round(1000 / float(np.percentile(itls_arr, 50)), 2)
    }
    
def measure_win_rate_vs_stage1(
    compressed_model,
    stage1_model_path: str,
    tokenizer,
    groq_api_key: str,
    model_judge: str,
    n_samples: int = 50,
    device: str = "cuda",
) -> dict:
    """
    Judge: compressed model vs Stage 1 QLoRA merged.
    Win rate from perspective compressed model
    """
    from fcs.finetune.judge import run_llm_judge
    
    stage1_model = AutoModelForCausalLM.from_pretrained(
        stage1_model_path,
        dtype=torch.float16,
        device_map="auto"
    )
    
    results = run_llm_judge(
        base_model=stage1_model,
        ft_model=compressed_model,
        tokenizer=tokenizer,
        groq_api_key=groq_api_key,
        n_samples=n_samples,
        device=device,
        model=model_judge,
    )
    
    return {
        "win_rate_vs_stage1": results["judge_win_rate"],
        "tie_rate_vs_stage1": results["judge_tie_rate"],
        "loss_rate_vs_stage1": results["judge_loss_rate"],
    }

def run_compress_eval(
    model,
    tokenizer,
    cfg: CompressConfig,
    method_name: str,
    compress_metadata: dict,
    stage1_model_path: str,
    groq_api_key: str = "",
    model_judge: str = "",
    run_judge: bool = True,
    device: Optional[str] = None
) -> dict:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    ecfg: EvalConfig = cfg.eval
    
    print("\n[1/4] Computing perplexity...")
    ppl = compute_perplexity(
        model=model,
        tokenizer=tokenizer,
        device=device,
    )
    
    print("\n[2/4] GSM8K accuracy...")
    gsm = compute_gsm8k(model, tokenizer, device=device)
    
    print("\n[3/4] Inference latency...")
    latency = measure_latency(
        model,
        tokenizer,
        prompt=ecfg.latency_prompt,
        max_new_tokens=ecfg.latency_max_new_tokens,
        warmup_runs=ecfg.latency_warmup_runs,
        benchmark_runs=ecfg.latency_benchmark_runs,
        device=device,
    )
    
    judge = {}
    if run_judge and groq_api_key:
        print(f"[4/4] Win rate vs Stage 1...")
        judge = measure_win_rate_vs_stage1(
            compressed_model=model,
            stage1_model_path=stage1_model_path,
            tokenizer=tokenizer,
            groq_api_key=groq_api_key,
            model_judge=model_judge,
            n_samples=ecfg.judge_samples,
            device=device,
        )
    else:
        print(f"[4/4] Skipping judge (no Groq key)")
    
    output_dir = cfg.model.output_dir
    size_mb = sum(
        f.stat().st_size for f in Path(output_dir).rglob("*") if f.is_file()
        if f.suffix in (".bin", ".safetensors")
    ) / 1024**2 # mb
    
    peak_mem_mb = torch.cuda.max_memory_allocated() / 1024*82
    
    full_results = {
        "method": method_name,
        **compress_metadata,
        **ppl,
        **gsm,
        **latency,
        **judge,
        "size_mb": round(size_mb, 2),
        "peak_inference_mem_mb": round(peak_mem_mb, 2),
    }
    
    out_path = Path(output_dir) / "results.json"
    with open(out_path, "w") as f:
        json.dump(full_results, f, indent=2)
    
    print(f"\n Result saved to {out_path}")
    return full_results