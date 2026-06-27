import os
import json
from re import A
import time
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from datasets import load_dataset, Dataset

from fcs.compress.config import CompressConfig, DistillConfig

def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 3.0,
    alpha: float = 0.7
) -> torch.Tensor:
    """
    Combined KL divergence + cross-entropy loss.
    
    alpha * KL(soft_student || soft_teacher) + (1 - alpha) * CE(student, labels)
    """
    B, T, C = student_logits.shape
    
    with torch.no_grad():
        soft_teacher = F.softmax(teacher_logits / temperature, dim=-1)
    
    soft_student = F.log_softmax(student_logits / temperature, dim=-1)
    
    kl_loss = F.kl_div(
        soft_student.view(B * T, C),
        soft_teacher.view(B * T, C),
        reduction="batchmean"
    ) * (temperature ** 2)
    
    shift_logits = student_logits[:, :-1, :].contiguous().view(-1, C)
    shift_labels = labels[:, 1:].contiguous().view(-1)
    shift_labels[shift_labels != -100] = 0
    
    ce_loss = F.cross_entropy(
        shift_logits,
        shift_labels,
        ignore_index=-100
    )
    
    return alpha * kl_loss + (1 - alpha) * ce_loss

class DistillationTrainer(Trainer):
    """
    Custom Trainer that override compute_loss for use distillation_loss.
    """
    def __init__(self, teacher_model, temperature: float, alpha: float, **kwargs):
        super().__init__(**kwargs)
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.alpha = alpha
        
        for param in self.teacher_model.parameters():
            param.requires_grad = False # freeze model
        
        self.teacher_model.eval() # non-activate dropout
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        
        student_outputs = model(**inputs)
        student_logits = student_outputs.logits
        
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )
            teacher_logits = teacher_outputs.logits
        
        loss = distillation_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            labels=labels,
            temperature=self.temperature,
            alpha=self.alpha
        )
        
        return (loss, student_outputs) if return_outputs else loss

def prepare_distill_dataset(
    tokenizer: AutoTokenizer,
    dataset_name: str,
    dataset_fraction: float,
    max_seq_length: int,
    seed: int = 42,
) -> dict:
    raw = load_dataset(dataset_name, split="train")
    
    if dataset_fraction < 1.0:
        n = int(len(raw) * dataset_fraction)
        raw = raw.select(range(n))
    
    def format_and_tokenize(example):
        text = (
            f"### Instruction:\n{example["instruction"]}\n\n"
            f"### Response:\n{example['output']}"
        )
        
        tokens = tokenizer(
            text,
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        ) # type: ignore
        
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens
    
    tokenized = raw.map(format_and_tokenize, remove_columns=raw.column_names,
                        batched=False, desc="Tokenizing for distillation")
    
    split = tokenized.train_test_split(test_size=0.05, seed=seed) #type: ignore
    return {"train": split['train'], "test": split['test']}

def run_distill(cfg: CompressConfig):
    """
    Sequence-level knowledge distillation.
    Teacher = QLoRA-merged model (Stage 1 winner).
    Student = smaller Llama-3.2-1B (or a model smaller than the teacher).
    
    NOTE: Even if the teacher and student are the same size, distillation remains useful
    as a form of label smoothing via soft targets—though the primary benefit
    arises when the student is smaller than the teacher.
    """
    dcfg: DistillConfig = cfg.distill
    output_dir = dcfg.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Loading teacher from {dcfg.teacher_model_path}...")
    teacher = AutoModelForCausalLM.from_pretrained(
        dcfg.teacher_model_path,
        dtype=torch.float16,
        device_map="auto"
    )
    teacher.eval()
    
    print(f"Loading student from {dcfg.student_model_path}...")
    student = AutoModelForCausalLM.from_pretrained(
        dcfg.student_model_path,
        dtype=torch.float16,
        device_map="auto"
    )
    student.config.use_cache = False
    
    tokenizer = AutoTokenizer.from_pretrained(dcfg.student_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Preparing distillation dataset...")
    dataset = prepare_distill_dataset(
        tokenizer=tokenizer,
        dataset_name=dcfg.dataset_name,
        dataset_fraction=dcfg.dataset_fraction,
        max_seq_length=dcfg.max_seq_length,
    )
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=dcfg.num_train_epochs,
        per_device_train_batch_size=dcfg.per_device_train_batch_size,
        per_device_eval_batch_size=dcfg.per_device_train_batch_size,
        gradient_accumulation_steps=dcfg.gradient_accumulation_steps,
        learning_rate=dcfg.learning_rate,
        fp16=dcfg.fp16,
        bf16=dcfg.bf16,
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        gradient_checkpointing=True,
        report_to="wandb",
        run_name="distill-llama3.2",
        ddp_find_unused_parameters=False,
    )
    
    # FIXME: if occur error use DataCollatorSeq2Seq
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding="longest",
        return_tensors="pt",
    ) 
    
    t0 = time.time()
    
    trainer = DistillationTrainer(
        teacher_model=teacher,
        temperature=dcfg.temperature,
        alpha=dcfg.alpha,
        model=student,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    trainer.train()
    train_time = time.time() - t0
    
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    total_size_mb = sum(
        f.stat().st_size for f in Path(output_dir).rglob("*") if f.is_file()
    ) / 1024**2
    
    peak_mem_mb = torch.cuda.max_memory_allocated() / 1024**2

    metadata = {
        "method": "sequence_distillation",
        "teacher": dcfg.teacher_model_path,
        "student": dcfg.student_model_path,
        "temperature": dcfg.temperature,
        "alpha": dcfg.alpha,
        "train_time_sec": round(train_time, 2),
        "checkpoint_size_mb": round(total_size_mb, 2),
        "peak_memory_mb": round(peak_mem_mb, 2),
    }
    
    with open(Path(output_dir) / "distill_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[Distill] done — {train_time/60:.1f} min, "
          f"size {total_size_mb:.1f} MB, peak mem {peak_mem_mb:.1f} MB")

    return output_dir, metadata