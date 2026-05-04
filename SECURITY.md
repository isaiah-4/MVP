# Security Configuration

## `COURTVISION_API_TOKEN`

Optional API authentication token for the FastAPI service.

- Local development: leave unset to disable request authentication.
- Production: set to a long random secret value.
- Accepted headers:
  - `Authorization: Bearer <token>`
  - `X-API-Token: <token>`
- When set, every route except `/health` requires a valid token.

## `COURTVISION_CORS_ORIGINS`

Comma-separated list of allowed browser origins for cross-origin requests.

- Local development default: `*`
- Production: set explicit HTTPS origins only, for example:
  - `https://app.example.com`
  - `https://app.example.com,https://admin.example.com`
- Do not use `*` in production.
