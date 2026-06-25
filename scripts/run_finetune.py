import os
from dotenv import load_dotenv

load_dotenv()
# ENV
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "")

if not GROQ_API_KEY:
    import groq
    raise groq.GroqError("GROQ_API_KEY not found")

if not MODEL_NAME:
    MODEL_NAME = "llama3-70b-8192"

import argparse
import json
import sys
from pathlib import Path

from numpy import full
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fcs.finetune.config import FinetuneConfig
from fcs.finetune.data import load_and_prepare_dataset
from fcs.finetune.trainer import train, load_tokenizer
from fcs.finetune.eval import run_eval

def parse_args():
    parser = argparse.ArgumentParser(description="Run one finetune experiment")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config, e.g. configs/finetune/lora.yaml"
    )
    
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Skip GSM8K eval (debug mode)"
    )
    
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="Skip LLM-as-judge eval (debug mode)"
    )
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Loading config from {args.config}")
    cfg = FinetuneConfig.from_yaml(args.config)
    
    tokenizer = load_tokenizer(cfg)
    
    print("Preparing dataset...")
    dataset = load_and_prepare_dataset(cfg.data, tokenizer, seed=42) #type: ignore
    print(f"    train: {len(dataset['train'])} | val: {len(dataset['val'])} | test: {len(dataset['test'])}")
    
    train_metrics = train(cfg, dataset)
    
    # reload model
    from transformers import AutoModelForCausalLM
    import torch
    
    base_model = None
    if not args.skip_judge:
        print("\nLoading base model for judge comparison...")
        base_model = AutoModelForCausalLM.from_pretrained(
            cfg.model.name_or_path,
            dtype=torch.float16,
            device_map="auto"
        )
    
    print("\nLoading fine-tuned model for checkpoint...")
    ft_model = AutoModelForCausalLM.from_pretrained(
        cfg.training.output_dir,
        dtype=torch.float16,
        device_map="auto"
    )
    
    full_metrics = run_eval( #type: ignore
        model=ft_model,
        base_model=base_model
        tokenizer=tokenizer, #type: ignore
        dataset=dataset['test'],
        cfg=cfg,
        train_metrics=train_metrics,
        groq_api_key=GROQ_API_KEY,
        run_judge=not args.run_judge,
        run_benchmark=not args.run_benchmark,
        model_judge=MODEL_NAME
    )
    
    print("\n" + "=" * 50)
    print(f"{'FINAL METRICS':^25}")
    print("=" * 50)
    for k, v in full_metrics.items():
        print(f"  {k:<30} {v}")
    print("=" * 50 + "\n")

if __name__ == "__main__":
    main()  