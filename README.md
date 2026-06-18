# Smart 3D Medical Disease Prediction

An AI-assisted clinical reasoning system combining a fine-tuned medical
language model, an imaging-based disease prediction model, and a
multi-agent orchestration layer, exposed through a backend API and
frontend client.

## System Overview

| Component | Description | Owner |
|---|---|---|
| `ai-models/fchat-llm` | LoRA fine-tuned LLaMA-2-7B for medical symptom assessment and structured clinical reporting | see `CODEOWNERS` |
| `ai-models/image-3d-model` | Imaging-based disease prediction model | see `CODEOWNERS` |
| `ai-models/multi-agent-system` | Agent orchestration layer coordinating the LLM and imaging model into a single workflow | see `CODEOWNERS` |
| `backend` | API layer exposing the AI pipeline to the frontend | see `CODEOWNERS` |
| `frontend` | Client application for symptom intake, imaging upload, and report display | see `CODEOWNERS` |

## Repository Layout

```
Smart-3D-Medical-Disease-Prediction/
  ai-models/
    fchat-llm/
    image-3d-model/
    multi-agent-system/
  backend/
  frontend/
  docs/
  .github/
    workflows/
  .gitignore
  README.md
  CODEOWNERS
  CONTRIBUTING.md
```

## Getting Started

Each top-level component is self-contained with its own `requirements.txt`
(or `package.json`) and `README.md`. Set up a component by entering its
directory and following its local README.

```
cd ai-models/fchat-llm
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Branching and Contribution Workflow

See `CONTRIBUTING.md` for branch naming conventions, commit message
format, and pull request expectations.

## Data and Model Weights

Datasets and trained model weights are not committed to this repository.
Each component's README documents where to obtain the relevant data and
how to point the code at a local copy. See the root `.gitignore` for the
excluded file patterns.

## License

This project is for academic/engineering purposes. The F-Chat language
model is derived from `meta-llama/Llama-2-7b-hf` and remains subject to
the Llama 2 Community License Agreement. Refer to each component's
README for any additional license terms.
