# Deploy weathermcp to Cloud Run

One container runs everything (`start.sh`): weather MCP (:8001) and quote MCP (:8002) on
localhost, agent + UI on `$PORT`. Visitors bring their own OpenRouter key (BYOK), so
hosting has no LLM cost.

## Prerequisites (once)

```bash
gcloud auth login
gcloud config set project <PROJECT_ID>
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
```

## Deploy

```bash
cd weathermcp-openaisdk
gcloud run deploy weathermcp \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --max-instances 1 \
  --memory 512Mi
```

| Flag | Why |
|---|---|
| `--allow-unauthenticated` | Public page; no server secret (key comes from the visitor) |
| `--max-instances 1` | Caps hosting cost even under bot traffic |
| `--memory 512Mi` | Three small Python processes fit comfortably |

No env vars needed. Optional: `--set-env-vars MODEL=<openrouter-model-id>` (must support tool calling).

## BYOK behaviour

- UI sends the key in the `X-OpenRouter-Key` header; it lives in page memory only.
- Server creates an OpenAI SDK client (base URL = OpenRouter) with it for that request only; never stored or logged. Missing → 401, invalid → 401 "invalid OpenRouter API key".
- Each question costs at most 5 LLM calls (`max_steps=4` + final answer); messages capped at 500 chars.

## Operate

- Logs: `gcloud run services logs read weathermcp --region us-central1`
- Delete: `gcloud run services delete weathermcp --region us-central1`

## Local

- Separate services: `docker compose up` (compose overrides each service's command).
- Single container like Cloud Run, without Docker: `PORT=8000 ./start.sh` (in a venv with `requirements.txt`).
