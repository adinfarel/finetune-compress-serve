import os
import time
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer #type: ignore
from datasets import DatasetDict
from wandb import config

from fcs.finetune.config import FinetuneConfig

def load_tokenizer(cfg: FinetuneConfig) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.name_or_path,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    return tokenizer

def load_model(cfg: FinetuneConfig) -> AutoModelForCausalLM:
    """
    Load model appropiate with method
    - full / lora : standard load in float16
    - qlora       : load with BitsAndBytesConfig (NF4 quantization)
    """
    method = cfg.training.method
    
    if method == "qlora":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None
        
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name_or_path,
        quantization_config=bnb_config,
        dtype=torch.float16,
        device_map="auto",
        attn_implementation=cfg.model.attn_implementation,
        trust_remote_code=True,
    )
    
    model.config.use_cache = False
    
    return model #type: ignore

def apply_lora(model, cfg: FinetuneConfig, is_qlora: bool):
    """
    Plugin LoRA adapters to model.
    For qlora, there is one more step: prepare_model_for_kbit_training.
    """
    if is_qlora:
        # cast layer norm to float32 and enable gradient checkpointing
        # so that backward pass can running over model quantized
        model = prepare_model_for_kbit_training(model)
    
    lora_cfg = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.lora_alpha,
        lora_dropout=cfg.lora.lora_dropout,
        target_modules=cfg.lora.target_modules,
        bias=cfg.lora.bias, #type: ignore
        task_type=cfg.lora.task_type,
    )
    
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    
    return model

def build_training_args(cfg: FinetuneConfig) -> TrainingArguments:
    return TrainingArguments(
        output_dir=cfg.training.output_dir,
        num_train_epochs=cfg.training.num_train_epochs,
        per_device_eval_batch_size=cfg.training.per_device_train_batch_size,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        warmup_ratio=cfg.training.warmup_ratio,
        fp16=cfg.training.fp16,
        bf16=cfg.training.bf16,
        logging_steps=cfg.training.logging_steps,
        save_steps=cfg.training.save_steps,
        eval_steps=cfg.training.eval_steps,
        evaluation_strategy="steps", #type: ignore
        load_best_model_at_end=True,
        report_to=cfg.training.report_to,
        run_name=cfg.training.run_name,
        gradient_checkpointing=True, 
        ddp_find_unused_parameters=False,
    )

def train(cfg: FinetuneConfig, dataset: DatasetDict) -> dict:
    """Entry point training.
    
    Return dict contain metrics for noted.
    """
    method = cfg.training.method
    print(f"\n{'='*50}")
    print(f"  Starting training - method: {method.upper()}")
    print(f"{'='*50}\n")
    
    # load model and tokenizer
    tokenizer = load_tokenizer(cfg)
    model = load_model(cfg)
    
    if method in ("lora", "qlora"):
        model = apply_lora(model, cfg, is_qlora=(method=="qlora"))
    
    training_args = build_training_args(cfg)
    
    # setup trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        tokenizer=tokenizer, #type: ignore
        max_seq_length=cfg.data.max_seq_length, #type: ignore
        dataset_text_field="text", #type: ignore
        packing=False, #type: ignore
    )
    
    torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    
    train_result = trainer.train()
    
    t_end = time.time()
    peak_vram = torch.cuda.max_memory_allocated() / 1024**3 # GB
    
    metrics = {
        "method": method,
        "train_loss": train_result.training_loss,
        "train_time_seconds": round(t_end - t_start, 2),
        "peak_vram_gb": round(peak_vram, 3),
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad), #type: ignore
        "total_params": sum(p.numel() for p in model.parameters()), #type: ignore
    }
    
    trainer.save_model(cfg.training.output_dir)
    tokenizer.save_pretrained(cfg.training.output_dir) #type: ignore
    
    print(f"\nDone. Peak VRAM: {peak_vram:.3f} GB | Time: {(t_end-t_start)/60:.1f} min")
    
    return metrics