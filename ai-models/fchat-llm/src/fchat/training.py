from __future__ import annotations

from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
    set_seed,
)

from . import config as cfg


def load_tokenizer() -> PreTrainedTokenizerBase:
    """Load the F-Chat tokenizer, preferring a local export over the base model."""
    required = ["tokenizer_json", "tokenizer_model", "tokenizer_config"]
    local_files_present = all(
        Path(cfg.TOKENIZER_DIR, cfg.TOKENIZER_FILES[k]).exists() for k in required
    )

    if local_files_present:
        tokenizer = AutoTokenizer.from_pretrained(cfg.TOKENIZER_DIR, use_fast=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(cfg.BASE_MODEL, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    return tokenizer


def tokenize_jsonl(
    jsonl_path: str,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = cfg.MAX_SEQ_LENGTH,
    num_proc: int = cfg.NUM_PROC,
) -> Dataset:
    """Tokenize a JSONL corpus for causal-LM training (labels = input_ids)."""
    raw_ds = load_dataset("json", data_files=jsonl_path, split="train")

    def _tokenize(batch):
        out = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        out["labels"] = out["input_ids"].copy()
        return out

    tokenized = raw_ds.map(
        _tokenize,
        batched=True,
        num_proc=num_proc,
        remove_columns=raw_ds.column_names,
    )
    return tokenized


def _load_base_model(model_path: str, tokenizer: PreTrainedTokenizerBase):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.resize_token_embeddings(len(tokenizer))
    return model


def build_lora_model(base_model):
    """Attach a fresh LoRA adapter to a base model using cfg.LORA_CONFIG."""
    base_model = prepare_model_for_kbit_training(base_model)
    lora_cfg = LoraConfig(**cfg.LORA_CONFIG)
    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()
    return model


def build_trainer(
    model,
    tokenized_dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    output_dir: str,
    training_args: dict,
) -> Trainer:
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    hf_training_args = TrainingArguments(output_dir=output_dir, **training_args)
    return Trainer(
        model=model,
        args=hf_training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )


def run_training_phase(
    phase_name: str,
    base_model_path: str,
    tokenized_dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    checkpoint_dir: str,
    output_dir: str,
    training_args: dict,
    mode: str = cfg.MODE,
) -> Trainer:
    """Run one curriculum phase: load base, attach LoRA, train, merge, save."""
    set_seed(training_args.get("seed", cfg.SEED))

    base_model = _load_base_model(base_model_path, tokenizer)
    model = build_lora_model(base_model) if mode == "lora" else base_model

    trainer = build_trainer(model, tokenized_dataset, tokenizer, checkpoint_dir, training_args)

    print(f"[{phase_name}] starting training (mode={mode})")
    if len(tokenized_dataset) > 0:
        trainer.train()
    print(f"[{phase_name}] training complete")

    _save_phase_output(trainer.model, tokenizer, output_dir, mode)
    return trainer


def _save_phase_output(model, tokenizer: PreTrainedTokenizerBase, output_dir: str, mode: str) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if mode == "lora":
        try:
            merged = model.merge_and_unload()
            merged.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"[SAVE] merged model + tokenizer saved to {output_dir}")
            return
        except Exception as exc:
            print(f"[ERROR] LoRA merge failed: {exc}")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[SAVE] model + tokenizer saved to {output_dir}")


def generate_sample(
    model,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> str:
    """Generate a short completion for prompt; used as a quick sanity check."""
    model = model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)
