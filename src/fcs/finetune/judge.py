import os
import time
from urllib import response
import torch
from typing import Optional
from transformers import PreTrainedModel, PreTrainedTokenizer, PretrainedBartModel
from datasets import load_dataset

JUDGE_SYSTEM_PROMPT = """\
You are an impartial judge evaluating the quality of two AI assistant responses.
Given an instruction and two responses (A and B), determine which response is better.
A better response is more helpful, accurate, complete, and well-structured.

Respond with ONLY one of these three options:
- "A" if response A is better
- "B" if response B is better  
- "tie" if they are equally good
"""

JUDGE_USER_TEMPLATE = """\
### Instruction:
{instruction}

### Response A:
{response_a}

### Response B:
{response_b}

Which response is better? Answer with only A, B, or tie.
"""

def generate_response(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    device: Optional[str] = None,
) -> str:
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(device) #type: ignore
    
    with torch.no_grad():
        outputs = model.generate( #type: ignore
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    return tokenizer.decode( 
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    ).strip() #type: ignore
    
def judge_llm_groq(
    instruction: str,
    response_a: str,
    response_b: str,
    groq_api_key: str,
    model: str = "llama3-70b-8192"
) -> str:
    """
    Send to GROQ API, return 'A', 'B' or 'tie'.
    """
    try:
        from groq import Groq
        client = Groq(api_key=groq_api_key)
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role", "user", "content": JUDGE_USER_TEMPLATE.format( #type: ignore
                    instruction=instruction,
                    response_a=response_a,
                    response_b=response_b
                )}, 
            ],
            temperature=0.0,
            max_tokens=10,
        )
        
        verdict = response.choices[0].message.content.strip().lower() #type: ignore
        if verdict in ("a", "b", "tie"):
            return verdict
        return "tie"

    except Exception as e:
        print(f"Judge API error: {e}")
        return "tie"

def run_llm_judge(
    base_model: PreTrainedModel,
    ft_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    groq_api_key: str,
    n_samples: int = 50,
    device: str = "cuda",
    model: str = "llama3-70b-8192"
) -> dict:
    """
    Running LLM-as-judge comparison.
    base_model = A, ft_model = B.
    Return win/tie/loss rate from subjective ft_model
    """
    
    dataset = load_dataset("tatsu-lab/alpaca", split=f"train[:{n_samples}]")
    
    wins, ties, losses = 0, 0, 0
    
    for i, example in dataset:
        instruction = example['instruction']
        input_ctx = example.get('inputs', '').strip()
        
        prompt = f"### Instruction:\n{instruction}"
        if input_ctx:
            prompt += f"\n\n### Input:\n{input_ctx}"
        
        prompt += f"\n\n### Response:\n"
        
        response_base = generate_response(base_model, tokenizer, prompt, device=device)
        response_ft = generate_response(ft_model, tokenizer, prompt, device=device)
        
        verdict = judge_llm_groq(
            instruction=instruction,
            response_a=response_base,
            response_b=response_ft,
            groq_api_key=groq_api_key,
            model=model
        )
        
        if verdict == "b":
            wins += 1
        elif verdict == "tie":
            ties += 1
        else:
            losses += 1
        
        time.sleep(0.5)
        
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{n_samples}] W:{wins} T:{ties} L:{losses}")
    
    total = wins + ties + losses

    return {
        "judge_win_rate": round(wins / total, 4),
        "judge_tie_rate": round(ties / total, 4),
        "judge_loss_rate": round(losses / total, 4),
        "judge_wins": wins,
        "judge_ties": ties,
        "judge_losses": losses,
        "judge_total": total,
    }