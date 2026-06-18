# Multi-Agent System

This folder hosts the orchestration layer that coordinates the F-Chat LLM,
the imaging model, and any rule-based or retrieval components into a single
clinical reasoning workflow (for example: an intake agent, a symptom
triage agent, an imaging-analysis agent, and a final report-generation
agent).

## Expected Structure

```
multi-agent-system/
  src/
    agents/
      __init__.py
      intake_agent.py
      triage_agent.py
      imaging_agent.py
      report_agent.py
    orchestrator.py
    config.py
  notebooks/
    agent_pipeline_demo.ipynb
  requirements.txt
  README.md
```

## Guidelines for the Owner of This Module

- Keep each agent in its own module with a single, well-defined
  responsibility and a documented input/output contract.
- The orchestrator should import the F-Chat inference interface from
  `ai-models/fchat-llm` and the imaging model's inference interface from
  `ai-models/image-3d-model` rather than duplicating model-loading code.
- Document the message/state schema passed between agents.
- Add a `README.md` here describing the agent framework used (if any),
  the workflow diagram, and how to run the orchestrator end to end.

## Status

Pending: implementation owned by the multi-agent team member.
