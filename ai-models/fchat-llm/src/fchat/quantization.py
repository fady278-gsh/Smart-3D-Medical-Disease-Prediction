from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Optional

from transformers import PreTrainedTokenizerBase

from . import config as cfg

try:
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

    HAS_GPTQ = True
except ImportError:
    HAS_GPTQ = False


def build_calibration_examples(
    tokenizer: PreTrainedTokenizerBase,
    jsonl_paths: List[str],
    n_samples: int = cfg.N_CALIBRATION_SAMPLES,
    max_length: int = cfg.MAX_SEQ_LENGTH,
    seed: int = cfg.SEED,
) -> List[dict]:
    """Sample and tokenize text records used to calibrate GPTQ quantization."""
    texts: List[str] = []
    for path in jsonl_paths:
        p = Path(path)
        if not p.exists():
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                try:
                    texts.append(json.loads(line)["text"])
                except (json.JSONDecodeError, KeyError):
                    continue

    if not texts:
        print("[GPTQ] no calibration text found in the provided JSONL files")
        return []

    rng = random.Random(seed)
    sample = rng.sample(texts, min(n_samples, len(texts)))

    examples = [
        tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        for text in sample
    ]
    return examples


def quantize_to_gptq(
    model_dir: str,
    quant_dir: str,
    tokenizer: PreTrainedTokenizerBase,
    calibration_examples: List[dict],
    gptq_config: Optional[dict] = None,
) -> bool:
    """Quantize the merged model to 4-bit GPTQ and save shards + tokenizer."""
    if not HAS_GPTQ:
        print("[GPTQ] auto-gptq is not installed. Install with: pip install auto-gptq")
        return False

    gptq_config = gptq_config or cfg.GPTQ_CONFIG

    if not calibration_examples:
        print("[GPTQ] no calibration examples available, skipping quantization")
        return False

    Path(quant_dir).mkdir(parents=True, exist_ok=True)

    try:
        quantize_config = BaseQuantizeConfig(**gptq_config)
        q_model = AutoGPTQForCausalLM.from_pretrained(model_dir, quantize_config)
        q_model.quantize(calibration_examples)
        q_model.save_quantized(quant_dir, use_safetensors=True)
        tokenizer.save_pretrained(quant_dir)
        print(f"[GPTQ] quantized model saved to {quant_dir}")
        return True
    except Exception as exc:
        print(f"[GPTQ] quantization failed: {exc}")
        return False


MODEL_CARD_TEMPLATE = """---
license: llama2
base_model: meta-llama/Llama-2-7b-hf
tags:
  - medical
  - clinical
  - lora
  - gptq
  - text-generation
language:
  - en
---

# F-Chat - Clinical Reasoning Language Model

F-Chat is a LoRA fine-tuned LLaMA-2-7B model specialized for medical
symptom assessment and structured clinical reporting.

## Training Curriculum

| Phase | Goal | Data |
|---|---|---|
| 1. Raw medical model | Domain-adaptive continued pre-training | RedPajama + PubMed abstracts (~16M) + EPFL clinical guidelines (~36k) |
| 2. F-Chat | Dialogue / instruction alignment | ~100k doctor-patient Q&A pairs |

Both phases use LoRA (r=16, alpha=32, target modules q_proj/v_proj,
dropout=0.05).

## Quantization

This repository hosts the 4-bit GPTQ post-training quantized export
(group size 128).

## Limitations and Intended Use

This model is a research/engineering prototype. It is not a certified
medical device and must not be used as a substitute for professional
medical advice, diagnosis, or treatment.

## Base Model License

Weights are derived from meta-llama/Llama-2-7b-hf and remain subject to
the Llama 2 Community License Agreement.
"""


def write_model_card(output_dir: str) -> str:
    """Write README.md (model card) into output_dir."""
    path = Path(output_dir) / "README.md"
    path.write_text(MODEL_CARD_TEMPLATE, encoding="utf-8")
    return str(path)
