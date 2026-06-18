# Backend

API layer that exposes the AI pipeline (F-Chat, imaging model, multi-agent
orchestrator) to the frontend client.

## Expected Structure

```
backend/
  app/
    main.py
    api/
      routes/
    core/
      config.py
    services/
      fchat_service.py
      imaging_service.py
      agent_service.py
    models/
      schemas.py
  requirements.txt
  Dockerfile
  README.md
```

## Guidelines for the Owner of This Module

- Keep service wrappers thin: each `*_service.py` should call into the
  corresponding module under `ai-models/` rather than reimplementing
  model logic.
- Document all endpoints (path, method, request/response schema).
- Use environment variables for secrets and model paths; do not hardcode
  them. Provide a `.env.example` file.
- Add a `README.md` here with setup instructions and a list of endpoints.

## Status

Pending: implementation owned by the backend team member.
