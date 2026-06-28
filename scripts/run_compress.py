import os
from dotenv import load_dotenv
from torch import quantization
load_dotenv()

# ENV
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "")

# assume this file in google colab
if not GROQ_API_KEY:
    try:
        from google.colab import userdata #type: ignore
        GROQ_API_KEY = userdata.get("GROQ_API_KEY")
    except ImportError as e:
        raise ImportError("Can't import") from e
    
if not MODEL_NAME:
    try:
        from google.colab import userdata #type: ignore
        MODEL_NAME = userdata.get("MODEL_NAME")
    except ImportError as e:
        raise ImportError("Can't import") from e

if not GROQ_API_KEY:
    import groq
    raise groq.GroqError("GROQ_API_KEY not found")

if not MODEL_NAME:
    MODEL_NAME = "llama3-70b-8192"

import argparse
import sys
import json
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fcs.compress.config import CompressConfig
from fcs.compress.quant import run_quantize
from fcs.compress.prune import run_prune
from fcs.compress.distill import run_distill
from fcs.compress.eval import run_compress_eval
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def _merged_win_stage1():
    from peft import PeftModel, AutoPeftModelForCausalLM
    merged = AutoPeftModelForCausalLM.from_pretrained(
        "results/stage1/qlora",
        dtype=torch.float16,
        device_map="auto"
    )
    merged = merged.merge_and_unload()
    
    merged_path = Path(CompressConfig.model.input_model_path)
    merged_path.mkdir(parents=True, exist_ok=True)
    
    merged.save_pretrained(merged_path, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained("results/stage1/qlora")
    tokenizer.save_pretrained(merged_path)
    print(f"Merged model saved to {str(merged_path)!r}")

def _validate_merged_exists():
    merged_path = Path(CompressConfig.model.input_model_path)
    
    if not merged_path.exists():
        raise FileNotFoundError("QLoRA merged directory not found, go first merged them.")

def parse_args():
    parser = argparse.ArgumentParser(description="Run one compress experiment.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    return parser.parse_args()

def load_model_for_eval(cfg: CompressConfig):
    """Load model compression for eval."""
    method = cfg.quant.method if cfg.method == "quant" else cfg.method
    output_dir = cfg.model.output_dir
    
    if method == "int8":
        bnb = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model.input_model_path,
            quantization_config=bnb,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.input_model_path)
    
    elif method == "int4_bnb":
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg.quant.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=cfg.quant.bnb_4bit_use_double_quant,
        )
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model.input_model_path,
            quantization_config=bnb,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.input_model_path)
    
    elif method == "awq":
        from awq import AutoAWQForCausalLM #type: ignore
        model = AutoAWQForCausalLM.from_quantized(
            output_dir,
            fuse_layers=True,
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(output_dir)
    
    else:
        model = AutoModelForCausalLM.from_pretrained(
            output_dir,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(output_dir)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def main():
    args = parse_args()
    
    _merged_win_stage1()
    assert _validate_merged_exists()
    
    print(f"Loading config from {args.config}")
    cfg = CompressConfig.from_yaml(args.config)
    method = cfg.method
    
    compress_metadata = {}
    
    if method == "quant":
        result = run_quantize(cfg)
        
        if isinstance(result, tuple):
            model, tokenizer = result
            compress_metadata = {"quant_method": cfg.quant.method}
        else:
            model, tokenizer = load_model_for_eval(cfg)
            awq_meta_path = Path(cfg.model.output_dir) / "quant_awq.json"
            if awq_meta_path.exists():
                with open(awq_meta_path) as f:
                    compress_metadata = json.load(f)
    
    elif method == "prune":
        output_dir, compress_metadata = run_prune(cfg)
        model, tokenizer = load_model_for_eval(cfg)
    
    elif method == "distill":
        output_dir, compress_metadata = run_distill(cfg)
        model, tokenizer = load_model_for_eval(cfg)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    torch.cuda.reset_peak_memory_stats()
    
    full_results = run_compress_eval(
        model=model,
        tokenizer=tokenizer,
        cfg=cfg,
        method_name=method if method != "quant" else cfg.quant.method,
        compress_metadata=compress_metadata,
        stage1_model_path=args.stage1_model_path,
        groq_api_key=GROQ_API_KEY,
        model_judge=MODEL_NAME,
        run_judge=not args.skip_judge,
    )
    
    print("\n" + "="*60)
    print(f"{'COMPRESS SUMMARY':^30}")
    print("="*60)
    for k, v in full_results.items():
        print(f"  {k:<40} {v}")
    print("="*55 + "\n")

if __name__ == "__main__":
    main()