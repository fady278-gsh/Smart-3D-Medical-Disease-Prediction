from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

# Training mode and reproducibility
MODE: str = "lora"
DEV_MODE: bool = True
SEED: int = 42

# Base model (frozen backbone for LoRA fine-tuning)
BASE_MODEL: str = "meta-llama/Llama-2-7b-hf"

TOKENIZER_DIR: str = "./"
TOKENIZER_FILES: Dict[str, str] = {
    "tokenizer_json": "tokenizer.json",
    "tokenizer_model": "tokenizer.model",
    "tokenizer_config": "tokenizer_config.json",
    "special_tokens_map": "special_tokens_map.json",
    "generation_config": "generation_config.json",
    "model_config": "config.json",
}

MAX_SEQ_LENGTH: int = 1024

DatasetRef = Union[str, Tuple[str, str]]

# Phase 1 sources: general fluency (RedPajama) + medical knowledge (PubMed,
# EPFL guidelines)
PHASE1_DATASETS: Dict[str, DatasetRef] = {
    "redpajama": ("togethercomputer/RedPajama-Data-1T", "arxiv"),
    "pubmed": "ncbi/pubmed",
    "guidelines": "epfl-llm/guidelines",
}

# Row caps used only when DEV_MODE is True
PHASE1_DEV_SAMPLES: Dict[str, int] = {
    "redpajama": 50,
    "pubmed": 100,
    "guidelines": 100,
}

# Sampling fractions used for a full run (None = take the full split)
PHASE1_FULL_FRACTIONS: Dict[str, Optional[float]] = {
    "redpajama": 0.0005,
    "pubmed": 0.01,
    "guidelines": None,
}

PHASE1_JSONL: str = "data/phase1_medical_base.jsonl"
PHASE1_OUTPUT_DIR: str = "F-Chat-Medical-Base"
PHASE1_CHECKPOINT_DIR: str = PHASE1_OUTPUT_DIR + "-Checkpoints"

PHASE1_TRAINING_ARGS: Dict = {
    "num_train_epochs": 1,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 8,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "fp16": True,
    "logging_steps": 20,
    "save_steps": 1000,
    "save_total_limit": 2,
    "warmup_steps": 200,
    "seed": SEED,
}

# Phase 2 source: doctor-patient Q&A pairs (local CSV, ~100k rows)
PHASE2_QA_CSV: str = "data/medical_qa_100k.csv"
PHASE2_QA_COLUMNS: Dict[str, str] = {"question": "question", "answer": "answer"}
PHASE2_DEV_SAMPLES: int = 200

PHASE2_JSONL: str = "data/phase2_medical_qa.jsonl"
OUTPUT_DIR: str = "F-Chat-Model"
CHECKPOINT_DIR: str = OUTPUT_DIR + "-Checkpoints"

PHASE2_TRAINING_ARGS: Dict = {
    "num_train_epochs": 2,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 8,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "fp16": True,
    "logging_steps": 20,
    "save_steps": 1000,
    "save_total_limit": 2,
    "warmup_steps": 200,
    "seed": SEED,
}

# LoRA adapter shared by both phases
LORA_CONFIG: Dict = {
    "r": 16,
    "lora_alpha": 32,
    "target_modules": ["q_proj", "v_proj"],
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM",
}

# GPTQ 4-bit post-training quantization
QUANT_DIR: str = "F-Chat-Model-GPTQ"

GPTQ_CONFIG: Dict = {
    "bits": 4,
    "group_size": 128,
    "desc_act": False,
}

N_CALIBRATION_SAMPLES: int = 128
MAX_SHARD_SIZE: str = "5GB"

NUM_PROC: int = max(1, (os.cpu_count() or 1) // 2)


def ensure_directories() -> None:
    """Create all data/output directories used by the pipeline."""
    for directory in (
        "data",
        PHASE1_OUTPUT_DIR,
        PHASE1_CHECKPOINT_DIR,
        OUTPUT_DIR,
        CHECKPOINT_DIR,
        QUANT_DIR,
    ):
        Path(directory).mkdir(parents=True, exist_ok=True)
