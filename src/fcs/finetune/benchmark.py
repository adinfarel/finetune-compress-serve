import re
import torch
from typing import Optional
from datasets import load_dataset
from transformers import PreTrainedModel, PreTrainedTokenizer

GSM8K_PROMPT = """\
Below is a math problem. Solve it step by step, then state the final answer as a number.

### Problem:
{question}

### Solution:
"""

def extract_number(text: str) -> Optional[float]:
    """
    Parse last number which appear in output model.
    GSM8K convention: final answer usually in end of response.
    """
    numbers = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text.replace(",", ""))
    if numbers:
        try:
            return float(numbers[-1])
        except ValueError:
            return None
    
    return None

@torch.no_grad()
def evaluate_gsm8k(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    n_samples: int = 100,
    max_new_tokens: int = 256,
    device: Optional[str] = None,
) -> dict:
    """
    Evaluate model in subset GSM8K.
    n_samples=100 enough make fair comparison at T4 without too long.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model.eval()
    
    dataset = load_dataset("openai/gsm8k", "main", split=f"test").take(n_samples)
    
    correct = 0
    total = 0
    
    for example in dataset:
        prompt = GSM8K_PROMPT.format(question=example['question']) #type: ignore
        
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(device)

        outputs = model.generate( #type: ignore
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
        
        generated = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        pred = extract_number(generated) #type: ignore
        gold = extract_number(example['answer']) #type: ignore
        
        if pred is not None and gold is not None and abs(pred - gold) < 1e-6:
            correct += 1
        total += 1
        
    accuracy = correct / total if total > 0 else 0.0
    
    return {
        "gsm8k_accuracy": round(accuracy, 4),
        "gsm8k_correct": correct,
        "gsm8k_total": total,
    }