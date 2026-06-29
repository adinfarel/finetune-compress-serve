from dataclasses import dataclass, field
from typing import Optional
import yaml

from fcs.finetune import benchmark

@dataclass
class ServeModelConfig:
    fp16_model_path: str = "results/stage1/qlora_merged"
    awq_model_path: str = "results/stage2/quant_awq"
    pruned_model_path: str = "results/stage2/prune_depth"
    dtype: str = "float16"

@dataclass
class BenchmarkConfig:
    prompts: list = field(default_factory=lambda: [
        "Explain the concept of machine learning in simple terms.",
        "Write a short story about a robot learning to paint.",
        "What are the main differences between Python and JavaScript?",
        "Describe the water cycle step by step.",
        "Give me three tips for improving productivity at work.",
    ])
    max_new_tokens: int = 128
    concurrency_levels: list = field(default_factory=lambda: [1, 4, 8, 16])
    warmup_runs: int = 3
    benchmark_runs: int = 20

@dataclass
class VllmConfig:
    gpu_memory_utilization: float = 0.90
    max_model_run: int = 1024
    dtype: str = "float16"
    quantization: Optional[str] = None
    
    temperature: float = 0.0
    max_tokens: int = 128

@dataclass
class ServeConfig:
    model: ServeModelConfig = field(default_factory=ServeModelConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    vllm: VllmConfig = field(default_factory=VllmConfig)
    
    experiment: str = "hf_eager"
    output_dir: str = "results/stage3"
    
    @classmethod
    def from_yaml(cls, path: str) -> "ServeConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(
            model=ServeModelConfig(**raw.get("model", {})),
            benchmark=BenchmarkConfig(**raw.get("benchmark", {})),
            vllm=VllmConfig(**raw.get("vllm", {})),
            experiment=raw.get("experiment", "hf_eager"),
            output_dir=raw.get("output_dir", "results/stage3"),
        )