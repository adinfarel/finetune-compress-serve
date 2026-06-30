import yaml

from dataclasses import dataclass, field
from typing import Optional

@dataclass
class CompressModelConfig:
    input_model_path: str = "results/stage1/qlora_merged"
    output_dir: str = "results/stage2"
    dtype: str = 'float16'

@dataclass
class QuantConfig:
    method: str = "int8"
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_use_double_quant: bool = True
    
    awq_bits: int = 4
    awq_group_size: int = 128
    awq_zero_point: bool = True
    
    gptq_bits: int = 4
    gptq_group_size: int = 128
    gptq_desc_act: bool = False
    gptq_damp_percent: float = 0.01

@dataclass
class PruneConfig:
    method: str = "depth"
    num_layers_to_remove: int = 4
    scoring_metrics: str = "magnitude"
    calibration_samples: int = 128

@dataclass
class DistillConfig:
    teacher_model_path: str = "results/stage1/qlora_merged"
    student_model_path: str = "meta-llama/Llama-3.2-1B" 
    dataset_name: str = "tatsu-lab/alpaca"
    dataset_fraction: float = 0.3
    max_seq_length: int = 512
    
    temperature: float = 3.0
    alpha: float = 0.7
    
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    output_dir: str = "results/stage2/distil"
    fp16: bool = True
    bf16: bool = False

@dataclass
class EvalConfig:
    batch_size: int = 8
    max_seq_length: int = 512
    gsm8k_samples: int = 100
    judge_samples: int = 50
    
    latency_warmup_runs: int = 3
    latency_benchmark_runs: int = 20
    latency_prompt: str = "Explain the concept of machine learning in simple terms."
    latency_max_new_tokens: int = 128

@dataclass
class CompressConfig:
    model: CompressModelConfig = field(default_factory=CompressModelConfig)
    quant: QuantConfig = field(default_factory=QuantConfig)
    prune: PruneConfig = field(default_factory=PruneConfig)
    distill: DistillConfig = field(default_factory=DistillConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    
    method: str = "quant"
    
    @classmethod
    def from_yaml(cls, path: str) -> "CompressConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(
            model=CompressModelConfig(**raw.get("model", {})),
            quant=QuantConfig(**raw.get("quant", {})),
            prune=PruneConfig(**raw.get("prune", {})),
            distill=DistillConfig(**raw.get("distill", {})),
            eval=EvalConfig(**raw.get("eval", {})),
            method=raw.get("method", ""),
        )