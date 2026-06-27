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
    
    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest", return_tensors="pt")
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
    n_samples: int = 100,
    max_new_tokens: int = 256,
    device: Optional[str] = None,
) -> dict:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = load_dataset("openai/gsm8k", "main", split=f"test", streaming=True).take(n_samples)
    model.eval()
    correct = 0
    
    for ex in ds:
        prompt = (
            f"Below is a math problem. Solve it step by step.\n\n"
            f"### Problem:\n{ex['question']}\n\n### Solution:\n"
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
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
    
    ttfts, itl = [], []
    
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
    
    return dict()