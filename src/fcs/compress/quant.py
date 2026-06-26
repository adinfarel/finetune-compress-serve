from math import e
import os
import json
import time
import torch
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from fcs.compress.config import CompressConfig, QuantConfig

def load_calibration_texts(num_samples: int = 128) -> list[str]:
    """Load small calibration set, reused for AWQ calibration."""
    from datasets import load_dataset
    
    ds = load_dataset("tatsu-lab/alpaca", split="train", streaming=True).take(num_samples)
    texts = [ex['instruction'] + " " + ex.get("output", "") for ex in ds]
    return texts

def quantize_bitsandbytes(cfg: CompressConfig):
    """
    NOTE: bitsandbytes does NOT produce a seperate quantized checkpoint
    on disk. Quantization happens at load-time via BitsAndBytesConfig.
    What save here is metadata + the eval results; the "artifact"
    for this method is: original fp16 checkpoint path + this config.
    """
    qcfg: QuantConfig = cfg.quant
    model_path = cfg.model.input_model_path
    output_dir = cfg.model.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    if qcfg.method == "int8":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    elif qcfg.method == "int4_bnb":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=qcfg.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=getattr(torch, qcfg.bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=qcfg.bnb_4bit_use_double_quant,
        )
    else:
        raise ValueError(f"Unknown bitsandbytes method: {qcfg.method}")

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto"
    )
    load_time = time.time() - t0
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    mem_bytes = torch.cuda.memory_allocated()
    mem_mb = mem_bytes / (1024**2)
    
    metadata = {
        "method": qcfg.method,
        "input_model_path": model_path,
        "load_time_sec": load_time,
        "peak_memory_mb": mem_mb,
        "note": "bitsandbytes quantization is load-time only; "
                " no seperate checkpoint is saved, only this metadata.",
    }
    with open(Path(output_dir) / "quant_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"[bitsandbytes:{qcfg.method}] loaded in {load_time:.2f}s, "
          f"peak mem {mem_mb:.2f} MB")
    
    return model, tokenizer

def quantize_awq(cfg: CompressConfig):
    """
    AWQ produces an actual quantized checkpoint on disk, unlike
    bitsandbytes. This is the format directly compatible with vLLM in stage 3
    """
    try:
        from awq import AutoAWQForCausalLM #type: ignore
    except ImportError as e:
        raise ImportError(
            "autowq not installed. Run: pip install autoawq"
        ) from e
    
    qcfg: QuantConfig = cfg.quant
    model_path = cfg.model.input_model_path
    output_dir = cfg.model.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    t0 = time.time()
    model = AutoAWQForCausalLM.from_pretrained(
        model_path,
        safetensors=True,
        device_map="cuda",
    )
    
    quant_config = {
        "zero_point": qcfg.awq_zero_point,
        "q_group_size": qcfg.awq_group_size,
        "w_bit": qcfg.awq_bits,
        "version": "GEMM", 
    }
    
    calib_texts = load_calibration_texts(num_samples=128)
    
    model.quantize(
        tokenizer,
        quant_config=quant_config,
        calib_data=calib_texts
    )
    quant_time = time.time() - t0
    
    model.save_quantized(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    total_size = sum(
        f.stat().st_size for f in Path(output_dir).rglob("*") if f.is_file()
    )
    
    size_mb = total_size / (1024**2)
    
    metadata = {
        "method": "awq",
        "input_model_path": model_path,
        "quant_time_sec": quant_time,
        "checkpoint_size_mb": size_mb,
        "calibration_samples": len(calib_texts),
        "quant_config": quant_config,
        "note": "AWQ checkpoint saved to disk, directly loadable by vLLM.",
    }
    
    with open(Path(output_dir) / "quant_awq.json", 'w') as f:
        json.dump(metadata, f, indent=2)
        
    print(f"[AWQ] quantized in {quant_time:.2f}s, "
        f"checkpoint size {size_mb:.1f} MB, saved to {output_dir}")

    return output_dir

def run_quantize(cfg: CompressConfig):
    method = cfg.quant.method
    print(f"=== Running quantization: {method} ===")
    
    if method in ("int8", "int4_bnb"):
        return quantize_bitsandbytes(cfg)
    elif method == "awq":
        return quantize_awq(cfg)
    else:
        raise ValueError(f"Unknown quant method: {method}")