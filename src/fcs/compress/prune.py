import os
import json
import time
import torch
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from fcs.compress.config import CompressConfig, PruneConfig

def score_layers_by_magnitude(model) -> list[tuple[int, float]]:
    """Score each decoder layer based on L2-norm average all its weight.
    Layer with lowest score = least important = remove candidate."""
    scores = []
    layers = model.model.layers
    
    for i, layer in enumerate(layers):
        total_norm = 0.0
        count = 0
        for param in layer.parameters():
            total_norm += param.data.float().norm(2).item()
            count += 1
        avg_norm = total_norm / max(count, 1)
        scores.append((i, avg_norm))

    return scores

def score_layers_by_activation(
    model,
    tokenizer,
    calibration_samples: int = 128,
    max_length: int = 256,
    device: str = "cuda",
) -> list[tuple[int, float]]:
    """Score each decoder layer based on L2-norm average output activation
    More accurate rather than magnitude but need forward pass
    use calibration data from alpaca."""
    ds = load_dataset("tatsu-lab/alpaca", split="train", streaming=True)
    texts = [
        ex["instruction"] + " " + ex.get("output", "")
        for ex in ds.take(calibration_samples)
    ]
    
    activation_norms = [[] for _ in range(len(model.model.layers))]
    hooks = []
    
    def make_hook(idx):
        def hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            norm = hidden.float().norm(2, dim=-1).mean().item()
            activation_norms[idx].append(norm)
        return hook
    
    for i, layer in enumerate(model.model.layers):
        h = layer.register_forward_hook(make_hook(i))
        hooks.append(h)
    
    model.eval()
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            ).to(device)
            model(**inputs)
    
    # cleanup hooks
    for h in hooks:
        h.remove()
    
    scores = [
        (i, float(np.mean(norms)) if norms else 0.0)
        for i, norms in enumerate(activation_norms)
    ]
    
    return scores

def remove_layers(model, layer_indices_to_remove: list[int]):
    """
    Remove layer from model.model.layers as a in-place.
    Update model config so that consistency.
    """
    indices_set = set(layer_indices_to_remove)
    original_layers = model.model.layers
    kept = [l for i, l in enumerate(original_layers) if i not in indices_set]
    
    import torch.nn as nn
    model.model.layers = nn.ModuleList(kept)
    
    model.config.num_hidden_layers = len(kept)
    
    return model

def run_prune(cfg: CompressConfig):
    """
    Depth pruning: remove N layer with importance lowest score.
    Save pruned model to disk as a full checkpoint.
    """
    pcfg: PruneConfig = cfg.prune
    model_path = cfg.model.input_model_path
    output_dir = cfg.model.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    n_layers_before = len(model.model.layers)
    print(f"Layers before pruning: {n_layers_before}")
    
    print(f"Scoring layers via: {pcfg.scoring_metrics}")
    t0 = time.time()
    
    if pcfg.scoring_metrics == "magnitude":
        scores = score_layers_by_magnitude(model)
    elif pcfg.scoring_metrics == "activation":
        scores = score_layers_by_activation(
            model,
            tokenizer,
            calibration_samples=pcfg.calibration_samples,
            device=device,
        )
    else:
        raise ValueError(f"Unknown scoring metrics: {pcfg.scoring_metrics}")
    
    score_time = time.time() - t0
    
    scores_sorted = sorted(scores, key=lambda x: x[1])
    layers_to_remove = [idx for idx, _ in scores_sorted[:pcfg.num_layers_to_remove]]
    layers_to_remove_sorted = sorted(layers_to_remove)
    
    print(f"Removing layers: {layers_to_remove_sorted}")
    print(f"Scores of removed layers: "
          f"{[round(s, 4) for i, s in scores if i in set(layers_to_remove)]}")
    
    t1 = time.time()
    model = remove_layers(model, layers_to_remove_sorted)
    prune_time = time.time() - t1
    
    n_layers_after = len(model.model.layers)
    print(f"Layers after pruning: {n_layers_after}")
    
    torch.cuda.reset_peak_memory_stats()
    _ = model(
        **tokenizer("test", return_tensors="pt").to(device),
        use_cache=False,
    )
    peak_mem_mb = torch.cuda.max_memory_allocated() / 1024**2
    
    print(f"Saving pruned model to {output_dir}...")
    t2 = time.time()
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    save_time = time.time() - t2
    
    total_size_mb = sum(
        f.stat().st_size for f in Path(output_dir).rglob("*") if f.is_file()
    ) / 1024**2
    
    metadata = {
        "method": "depth_pruning",
        "scoring_metric": pcfg.scoring_metrics,
        "layers_before": n_layers_before,
        "layers_after": n_layers_after,
        "layers_removed": layers_to_remove_sorted,
        "num_layers_removed": pcfg.num_layers_to_remove,
        "score_time_sec": round(score_time, 2),
        "prune_time_sec": round(prune_time, 2),
        "save_time_sec": round(save_time, 2),
        "checkpoint_size_mb": round(total_size_mb, 2),
        "peak_memory_mb": round(peak_mem_mb, 2),
        "layer_scores": scores,
    }
    
    with open(Path(output_dir) / "prune_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"[Prune] done - {n_layers_before}->{n_layers_after} layers. "
          f"size {total_size_mb:.1f} MB, peak mem {peak_mem_mb:.1f} MB")
    
    return output_dir, metadata