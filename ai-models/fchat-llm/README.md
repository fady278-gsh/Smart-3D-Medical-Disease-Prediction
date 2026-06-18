# F-Chat

Two-stage curriculum training pipeline for F-Chat, a LoRA fine-tuned
LLaMA-2-7B model specialized for medical symptom assessment and structured
clinical reporting.

## Training Curriculum

| Phase | Goal | Data |
|---|---|---|
| 1. Raw medical model | Domain-adaptive continued pre-training | RedPajama (general fluency) + PubMed abstracts (~16M) + EPFL clinical guidelines (~36k) |
| 2. F-Chat | Dialogue / instruction alignment | ~100k doctor-patient Q&A pairs |

Both phases fine-tune with LoRA on top of `meta-llama/Llama-2-7b-hf`. The
final checkpoint is merged and post-training quantized to 4-bit with GPTQ.

## Project Layout

```
F-Chat-Model/
  src/fchat/
    config.py          training configuration and dataset references
    data_pipeline.py    dataset streaming, cleaning, JSONL construction
    training.py         tokenizer, LoRA model, trainer, generation
    quantization.py     GPTQ calibration, quantization, model card
  notebooks/
    fchat_pipeline.ipynb end-to-end orchestration notebook
  data/                 generated JSONL corpora (gitignored)
  requirements.txt
  pyproject.toml
```

## Setup

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Open `notebooks/fchat_pipeline.ipynb` and run cells top to bottom. The
notebook is split into the following stages:

1. Environment setup and configuration
2. Phase 1 data preparation (RedPajama, PubMed, EPFL guidelines)
3. Phase 1 training (raw medical model)
4. Phase 2 data preparation (doctor-patient Q&A)
5. Phase 2 training (F-Chat)
6. Evaluation / sanity-check generation
7. GPTQ quantization
8. Model card and export

Set `fchat.config.DEV_MODE = False` before running for a full-scale
training run; `DEV_MODE = True` (default) streams a small sample of each
dataset for end-to-end smoke testing.

## License

Base model weights are derived from `meta-llama/Llama-2-7b-hf` and remain
subject to the Llama 2 Community License Agreement.
