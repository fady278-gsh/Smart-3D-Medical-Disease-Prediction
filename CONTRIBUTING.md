# Contributing

## Branching

Create a feature branch off `main` for any change, named after your
component and the change:

```
feature/fchat-add-evaluation-metrics
feature/imaging-resnet50-training
feature/agents-triage-workflow
fix/backend-cors-config
```

Do not commit directly to `main`.

## Commits

Use short, descriptive, imperative commit messages:

```
Add GPTQ calibration step to fchat-llm pipeline
Fix tokenizer padding in image-3d-model preprocessing
```

## Pull Requests

1. Push your feature branch and open a pull request against `main`.
2. Fill in a short description of what changed and why.
3. CODEOWNERS will be requested automatically as reviewers based on the
   paths you touched.
4. Resolve review comments before merging. Squash-merge is preferred to
   keep `main` history clean.

## Adding a New Component

If your work doesn't fit an existing folder (`ai-models/*`, `backend`,
`frontend`), open an issue first to agree on where it belongs before
adding a new top-level directory.

## Data and Secrets

Never commit datasets, model weights, API keys, or `.env` files. Add new
file patterns to the root `.gitignore` if your component introduces a new
type of generated/secret file.
