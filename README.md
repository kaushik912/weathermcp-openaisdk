# Weather Agent (OpenAI SDK + MCP)

A small, provider-agnostic AI agent in the browser. Ask a question; the LLM decides
whether to call live tools (weather, quotes) exposed by **MCP servers**, then answers.
The UI shows every tool call the agent made.

- **Model layer:** OpenAI SDK against any OpenAI-compatible endpoint — OpenRouter by
  default (hundreds of models), or a local Ollama model, via one env var.
- **Tool layer:** two independent MCP servers (weather via [open-meteo](https://open-meteo.com),
  quotes). Tools are discovered at runtime, not hardcoded.
- **Agent loop:** hand-written (~60 lines): step budget, repeat-call detection, forced
  final answer — robust even with small models.
- **BYOK:** visitors paste their own OpenRouter key; it stays in the tab and is sent per
  request, never stored or logged. Hosting costs nothing.

See [`idea.md`](idea.md) for the full walkthrough. A LiteLLM-based variant lives in
[`ai-mcp-lab/weathermcp-litellm`](https://github.com/kaushik912/ai-mcp-lab/tree/master/weathermcp-litellm).

```
Browser ──X-OpenRouter-Key──▶ FastAPI /chat ──▶ agent loop ──▶ OpenRouter (OpenAI SDK)
                                                   │
                                                   ├──▶ weather MCP (:8001) ──▶ open-meteo
                                                   └──▶ quotes  MCP (:8002)
```

## Run locally

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
PORT=8000 ./start.sh          # both MCP servers + agent; open http://localhost:8000
```

Paste an OpenRouter key ([get one](https://openrouter.ai/keys)) in the UI.
Free local model instead: `LLM_BASE_URL=http://localhost:11434/v1 MODEL=llama3.1 PORT=8000 ./start.sh` (no key needed).

## Configuration

| Env var | Default | Notes |
|---|---|---|
| `LLM_BASE_URL` | `https://openrouter.ai/api/v1` | Any OpenAI-compatible endpoint |
| `MODEL` | `~deepseek/deepseek-v4-flash-latest` | Must support tool calling |
| `MCP_URLS` | from `mcp_config.json` | Comma-separated override |

## Deploy

Cloud Run, single container, scale-to-zero, max 1 instance — see [`deploy.md`](deploy.md).
