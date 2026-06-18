# Frontend

Client application consuming the backend API to present symptom intake,
imaging upload, and the resulting clinical report to the end user.

## Expected Structure

```
frontend/
  src/
    pages/
    components/
    services/
      api.ts
  public/
  package.json
  README.md
```

## Guidelines for the Owner of This Module

- Centralize all backend API calls in `src/services/api.ts` (or
  equivalent) rather than scattering fetch calls across components.
- Document required environment variables (API base URL, etc.) in a
  `.env.example` file.
- Add a `README.md` here with setup and run instructions.

## Status

Pending: implementation owned by the frontend team member.
