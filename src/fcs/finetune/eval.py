import math
import json
import torch
from pathlib import Path
from typing import Optional

from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer, DataCollatorWithPadding, PretrainedBartModel
from datasets import Dataset

from fcs.finetune.config import FinetuneConfig

@torch.no_grad()
def compute_perplexity(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    batch_size: int = 8,
    max_seq_length: int = 512,
    device: Optional[str] = None,
) -> dict:
    """
    Calculate perplexity model over given dataset.
    
    Return dict:
        perplexity: float
        avg_loss  : float (cross-entropy, before exp())
        n_tokens  : int (total token that evaluate)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model.eval()
    model.to(device) #type: ignore
    
    collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding="longest",
        return_tensors="pt",
    )
    
    loader = DataLoader(
        dataset, #type: ignore
        batch_size=batch_size,
        collate_fn=collator,
        shuffle=False
    )
    
    total_loss = 0.0
    total_tokens = 0
    
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        labels[labels == tokenizer.pad_token_id] = -100
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        
        n_tokens = (labels != -100).sum().item()
        total_loss += outputs.loss.item() * n_tokens
        total_tokens += n_tokens
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    return {
        "perplexity": round(perplexity, 4),
        "avg_loss": round(avg_loss, 6),
        "n_tokens": total_tokens,
    }

def save_results(metrics: dict, output_dir: str, filename: str = "results.json"):
    """
    Save metrics to JSON in output_dir
    Called after training and eval finish.
    """
    path = Path(output_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, mode='w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Results saved to {str(path)!r}")

def run_eval(
    model: PreTrainedModel,
    base_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    cfg: FinetuneConfig,
    train_metrics: dict,
    groq_api_key: str,
    run_judge: bool = True,
    run_benchmark: bool = True,
    model_judge: str = "llama3-70b-8192"
) -> dict:
    """
    Running full eval after training done.
    perplexity + GSM8K + LLM judge win rate.
    """
    from fcs.finetune.benchmark import evaluate_gsm8k
    from fcs.finetune.judge import run_llm_judge
    print("\nRunning evaluation on test...")
    
    print("\n[1/3] Computing perplexity...")
    ppl_metrics = compute_perplexity(
        model=model,
        dataset=dataset,
        tokenizer=tokenizer,
        batch_size=8,
        max_seq_length=cfg.data.max_seq_length,
    )
    
    gsm_metrics = {}
    if run_benchmark:
        print("\n[2/3] Computing GSM8K accuracy...")
        gsm_metrics = evaluate_gsm8k(model, tokenizer, n_samples=100)
    
    judge_metrics = {}
    if run_judge and base_model is not None:
        print("\n[3/3] Running LLM-as-judge...")
        judge_metrics = run_llm_judge(
            base_model=base_model,
            ft_model=model,
            tokenizer=tokenizer,
            groq_api_key=groq_api_key,
            n_samples=50,
            model=model_judge
        )
    
    full_metrics = {
        **train_metrics,
        "test_perplexity": ppl_metrics["perplexity"],
        "test_avg_loss": ppl_metrics["avg_loss"],
        "test_n_tokens": ppl_metrics["n_tokens"],
        "trainable_param_pct": round(
            train_metrics["trainable_params"] / train_metrics["total_params"] * 100, 4
        ),
        **gsm_metrics,
        **judge_metrics,
    }
    
    save_results(full_metrics, cfg.training.output_dir)
    
    return full_metrics