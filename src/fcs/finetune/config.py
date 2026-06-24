from dataclasses import dataclass, field
from typing import Optional
import yaml

@dataclass
class ModelConfig:
    name_or_path: str = "meta-llama/Llama-3.2-1B"
    torch_dtype: str = "float16"
    attn_implementation: str = "eager"
    
@dataclass
class DataConfig:
    dataset_name: str = "tatsu-lab/alpaca"
    train_split: str = "train"
    max_seq_length: int = 512
    dataset_fraction: float = 1.0

@dataclass
class LoraConfig:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list = field(default_factory=lambda: ["q_proj", "o_proj"])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

@dataclass
class TrainingConfig:
    method: str = "lora"
    output_dir: str = "results/stage1"
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    fp16: bool = True      
    bf16: bool = False
    report_to: str = "wandb"
    run_name: str = "lora-llama3.2-1b"

@dataclass
class FinetuneConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    @classmethod
    def from_yaml(cls, path: str) -> "FinetuneConfig":
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        
        return cls(
            model=ModelConfig(**raw.get("model", {})),
            data=DataConfig(**raw.get("data", {})),
            lora=LoraConfig(**raw.get("lora", {})),
            training=TrainingConfig(**raw.get("training", {})),
        )