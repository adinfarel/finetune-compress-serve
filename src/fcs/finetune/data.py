from typing import Optional
from datasets import load_dataset, DatasetDict
from transformers import PreTrainedTokenizer

from fcs.finetune.config import DataConfig

# prompt template

ALPACA_PROMPT_TEMPLATE = """\
Below is an instruction that describes a task{input_block}.
Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{output}"""

def format_alpaca_prompt(example: dict) -> str:
    """Convert one row Alpaca dataset become one string prompt"""
    input_block = (
        f", paired with an input that provides further context.\n\n### Input:\n{example["input"]}"
        if example.get("input", "").strip()
        else ""
    )
    return ALPACA_PROMPT_TEMPLATE.format(
        input_block=input_block,
        instruction=example["instruction"],
        output=example["output"],
    )

def load_and_prepare_dataset(
    cfg: DataConfig,
    tokenizer: PreTrainedTokenizer,
    seed: int = 42,
) -> DatasetDict:
    """
    Load Alpaca, format the prompt, tokenize, and then 
    split into train, validation, and test sets.
    
    Split ratio: 90% training / 5% validation / 5% testing
    The split is fixed using a seed to ensure consistency across all three experiments.
    """
    
    raw = load_dataset(cfg.dataset_name, split=cfg.train_split)
    
    if cfg.dataset_fraction < 1.0:
        n = int(len(raw) * cfg.dataset_fraction)
        raw = raw.select(range(n))
    
    def apply_template(example):
        return {"text": format_alpaca_prompt(example=example)}
    
    raw = raw.map(apply_template, remove_columns=raw.column_names)
    
    def tokenize(example):
        tokens = tokenizer(
            example['text'],
            truncation=True,
            max_length=cfg.max_seq_length,
            padding=False
        )
        
        tokens['labels'] = tokens["input_ids"].copy()
        return tokens

    tokenized = raw.map(tokenize, remove_columns=["text"], batched=True, batch_size=1000, desc="Tokenizing")
    
    train_valtest = tokenized.train_test_split(test_size=0.10, seed=seed)
    val_test = train_valtest["test"].train_test_split(test_size=0.50, seed=seed)
    
    return DatasetDict({
        "train": train_valtest["train"],
        "test": val_test["test"],
        "val": val_test["train"]
    })