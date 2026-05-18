# Mister Fritz — Discord AI Chatbot

Mister Fritz is an AI-powered Discord bot with a sophisticated, sardonic personality modeled after an English butler. It uses LangChain, LangGraph, and a locally-installed Ollama instance to provide conversational responses with persistent per-user memory, web search, document retrieval, image generation, voice synthesis, and more.

---

## Features

| Feature | Details |
|---|---|
| Conversational AI | Witty butler-like responses via local Ollama LLMs |
| Per-user Memory | Conversation summaries stored and retrieved from ChromaDB |
| Document RAG | Query local Word/PDF documents with `/lore` |
| Web Integration | Live web search (DuckDuckGo) and website scraping |
| Image Generation | Text-to-image via Stable Diffusion XL (GPU-accelerated, long-prompt capable) |
| Voice Synthesis | Text-to-speech via Coqui XTTS v2 (plays in voice channel or uploads as file) |
| Vision Analysis | Analyze user-attached images via LLaVA |
| File Operations | Sandboxed file read/write/edit/search/execute for the configured root user |
| Card Game | Built-in deck system with success/failure mechanics for tabletop-style play |
| Observability | Prometheus metrics on `:8000/metrics`, health on `:8000/health`, pre-built Grafana dashboard |
| Canary deployment | Argo Rollouts canary steps with Prometheus-based promotion gates |
| Local DevOps sim | Full canary pipeline simulation via Docker Compose — no cloud cluster required |

---

## Discord Slash Commands

| Command | Description |
|---|---|
| `/hello` | Greet the bot |
| `/lore <query>` | Search local RAG documents |
| `/gen <prompt>` | Generate an image from a text description |
| `/voice <message>` | Synthesize speech (plays in voice channel or uploads file) |
| `/join` / `/leave` | Join or leave the current voice channel |
| `/draw [n]` | Draw cards from a deck |
| `/cards_remaining` | Check how many cards remain |
| `/reload_deck` | Reset your deck |
| `/health` | Show system health and metrics |
| `/help`, `/about` | Discover bot capabilities and version info |
| `/workspace enable` | Create a sandboxed workspace and turn on file tools for yourself |
| `/workspace disable` / `status` | Manage your workspace |
| `/workspace set <path>` | (Admin only) Register an arbitrary host path as your workspace |
| `/forget memories` / `conversation` / `schedules` / `all` | Delete stored data Fritz has about you |
| `/export` | Download a JSON snapshot of everything Fritz has stored about you |

Direct messages or `@mentions` trigger the full agent with all tools.
Attach images or Discord voice messages — Fritz can analyze both.

---

## System Requirements

Mister Fritz runs the LLMs locally via Ollama, so hardware is the dominant cost.

| Model | Approx VRAM | Approx RAM (CPU mode) | Notes |
|---|---|---|---|
| `gpt-oss` (thinking model) | ~14 GB | ~20 GB | Primary reasoning model. CPU inference is usable but slow (~10 s/token range). |
| `llama3.2` (fast model) | ~4 GB | ~6 GB | Used for planning, summarisation, memory extraction. CPU mode is comfortable. |
| `llava` (vision) | ~5 GB | ~8 GB | Only loaded when an image is analysed. |
| `mxbai-embed-large` (embeddings) | ~1 GB | ~2 GB | Always loaded for RAG and memory. |
| Stable Diffusion XL (image gen) | ~8 GB | not viable on CPU | First run downloads ~7 GB of weights. |
| Coqui XTTS v2 (TTS) | ~2 GB | ~4 GB | First run downloads ~2 GB of weights. |

**Recommended baseline for the full feature set:**
- GPU with ≥ 16 GB VRAM (RTX 4080/4090, or two smaller GPUs)
- 32 GB system RAM
- 20 GB free disk for models and Chroma data

**Minimum for text-only operation (no image gen, no GPU):**
- 16 GB RAM
- Use `FAST_OLLAMA_MODEL` for both thinking and fast roles (set both env vars to `llama3.2`)
- Disable image generation by not pulling Stable Diffusion

---

## Prerequisites

### 1. Python 3.12+
Download from [python.org](https://www.python.org/downloads/).

### 2. Ollama (required — run natively on your host)
Ollama is **not** included in Docker Compose. Install it directly:

| Platform | Installation |
|---|---|
| Windows | Download from [ollama.com/download/windows](https://ollama.com/download/windows) |
| macOS | `brew install ollama` or [ollama.com/download/mac](https://ollama.com/download/mac) |
| Linux | `curl -fsSL https://ollama.com/install.sh \| sh` |

After installing, pull the required models:
```bash
ollama create -f modelfiles/gpt-oss-20b-modelfile.txt gpt-oss
ollama create -f modelfiles/llama3.2-modelfile.txt llama3.2
ollama create -f modelfiles/llava-modelfile.txt llava

ollama run gpt-oss /bye
ollama run llama3.2 /bye
ollama run llava /bye

ollama pull mxbai-embed-large
```

### 3. FFmpeg (for voice features)
- **Windows**: The repo includes `ffmpeg.exe` / `ffprobe.exe` — no extra install needed for native runs. Docker builds install FFmpeg via `apt-get`.
- **macOS/Linux**: `brew install ffmpeg` or `apt-get install -y ffmpeg`

### 4. PyTorch with CUDA (optional — for GPU-accelerated image generation and TTS)
```bash
# CUDA (Windows/Linux)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU-only / macOS
pip install torch torchvision torchaudio
```

---

## Installation

### Option A — Run natively

```bash
git clone <repository-url>
cd MisterFritz

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

**Configure the bot.** The fastest path is the interactive setup wizard, which
pings Ollama, creates/pulls the four required models, validates your Discord
token against the Discord API, and writes a complete `.env` for you:

```bash
python scripts/setup.py
```

It's idempotent — safe to re-run if you only need to update one value. It uses
only the Python standard library so you can run it before `pip install`.

If you'd rather configure by hand, copy the template and fill in the required
fields:

```bash
cp .env.example .env
```

```dotenv
# .env
DISCORD_BOT_TOKEN=your_token_here
ROOT_USER=your_discord_username
OLLAMA_HOST=http://127.0.0.1:11434
```

> All config previously in `config.json` is now read from environment variables.
> `config.json` still works as a fallback, but `.env` is the preferred approach.

**Start the bot:**
```bash
python main_discord.py
```

---

### Option B — Docker Compose (app + monitoring)

Ollama must still be running natively on your host. The compose stack runs the bot itself plus Prometheus and Grafana.

```bash
cp .env.example .env   # fill in DISCORD_BOT_TOKEN and ROOT_USER

docker compose up -d
```

| Service | URL |
|---|---|
| Grafana dashboards | http://localhost:3000 (default login: `admin` / `admin`) |
| Prometheus | http://localhost:9090 |

> On Linux, set `OLLAMA_HOST=http://172.17.0.1:11434` in `.env` (Docker bridge IP) or add `--add-host=host.docker.internal:host-gateway` to your compose config.

---

## Configuration Reference

All settings can be set as environment variables or in a `.env` file. See `.env.example` for the full list.

| Variable | Default | Description |
|---|---|---|
| `DISCORD_BOT_TOKEN` | — | **Required.** Your Discord bot token |
| `ROOT_USER` | — | **Required.** Discord username with admin privileges (`/workspace set`, `/schedule add/remove`) |
| `ADMIN_USERS` | — | Comma-separated additional admin usernames. Anyone listed gets the same powers as `ROOT_USER`. |
| `WORKSPACES_ROOT` | `./workspaces` | Parent directory for per-user sandboxed workspaces created via `/workspace enable`. |
| `AUDIT_LOG_PATH` | `audit.log` | Path to the append-only NDJSON audit log written on every `/forget` and `/export` event. |
| `ADMIN_PANEL_PASSWORD` | — | Set to enable the read-only web admin panel. Leave unset to disable. |
| `ADMIN_PANEL_PORT` | `8001` | Port for the admin panel. Bound to `127.0.0.1` only — use SSH port forwarding for remote access. |
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama API endpoint |
| `THINKING_OLLAMA_MODEL` | `gpt-oss` | Primary reasoning model |
| `FAST_OLLAMA_MODEL` | `llama3.2` | Fast model for quick tasks |
| `EMBEDDING_MODEL` | `mxbai-embed-large` | Embedding model for ChromaDB |
| `VISION_MODEL` | `llava` | Vision model for image analysis |
| `DOC_STORAGE_DESCRIPTION` | `anything you don't know about.` | Hint for when the agent queries documents |
| `DOC_FOLDER` | `./input` | Directory watched for RAG documents |
| `CHROMA_DB_PATH` | `./chroma_store` | ChromaDB persistence path |
| `FFMPEG_PATH` | auto-detected | Override FFmpeg binary path |

---

## Architecture

```
User (Discord)
    │
    ▼
main_discord.py   ──  slash commands, on_message, streaming handler
    │
    ▼
mister_fritz.py   ──  LangGraph state machine
    ├─ CONVERSATION_NODE   ──  ReAct agent, tool dispatch
    │   ├─ get_current_time, roll_dice
    │   ├─ scrape_web, search_web
    │   ├─ search_documents  →  document_engine  →  Chroma RAG
    │   ├─ search_memories   →  chroma_store     →  Chroma KV
    │   ├─ generate_image    →  image_generator  →  SDXL
    │   ├─ analyze_image     →  Ollama LLaVA
    │   └─ file tools (root only) →  file_tools.py
    └─ SUMMARIZE_NODE   ──  auto-summarise at 15+ messages, store to Chroma
```

Conversation state is checkpointed per-user in `chat_history.db` (SQLite).

---

## Testing

```bash
pip install pytest pytest-cov pytest-asyncio
pytest tests/ -v --cov=. --cov-report=term-missing
```

The test suite covers:
- `fritz_utils` — config loading, env var overrides, `validate_config`
- `sqlite_store` — CRUD, namespace isolation, prefix search
- `deck_of_cards` — Card logic, Deck, draw/reload/remaining
- `file_tools` — all 6 tools, authorization, path traversal prevention, `execute_command` allowlist + sandbox rejections
- `observability` — counters, latency rolling window, error tracking, health text
- `agent_tools` — `get_current_time`, `format_prompt`, `scrape_web` (mocked httpx), `search_web` (mocked), `roll_dice`
- `discord_commands` — `split_into_chunks`, `StreamingMessageHandler` rate limiting
- `bot_commands` — ROOT_USER gating on `schedule_add` / `schedule_remove`, open access on `schedule_list`
- `mister_fritz` — planner JSON parsing (code fences, surrounding text, malformed input, exception fallback)
- `document_engine` — `VECTORSTORE_LOCK` held during ingest, `_SHUTDOWN_SENTINEL` exits worker, `shutdown()` idempotency

CI runs on every push via GitHub Actions (see `.github/workflows/ci.yml`).

---

## Observability

The bot exposes a metrics server on port `8000` (configurable via `METRICS_PORT`):

| Endpoint | Description |
|---|---|
| `GET :8000/health` | JSON health snapshot (uptime, error rate, p99 latency) |
| `GET :8000/metrics` | Prometheus text format |

### Available Prometheus metrics

| Metric | Type | Labels |
|---|---|---|
| `misterfritz_discord_messages_total` | Counter | — |
| `misterfritz_tool_calls_total` | Counter | `tool` |
| `misterfritz_errors_total` | Counter | `operation` |
| `misterfritz_request_duration_seconds` | Histogram | `operation` |
| `misterfritz_uptime_seconds` | Gauge | — |

The compose stack in `docker-compose.yml` includes Prometheus and Grafana with a pre-provisioned dashboard.

---

## Admin panel

Set `ADMIN_PANEL_PASSWORD` in `.env` and the bot will start a read-only HTML admin panel at `http://127.0.0.1:8001/` (port configurable via `ADMIN_PANEL_PORT`). It's bound to localhost — for remote access, SSH-forward the port instead of exposing it.

Auth is HTTP Basic; any username works, only the password matters. Pages:

| Path | Purpose |
|---|---|
| `/` | Bot version, uptime, counters, error breakdown, latency summary |
| `/users` | Every user with stored data and counts of memories / schedules / workspace |
| `/users/<id>` | Per-user detail page — memories, schedules, conversation checkpoint count |
| `/schedules` | All scheduled tasks across all users |
| `/documents` | Files in `DOC_FOLDER` with sizes and modified times |
| `/health` | Same JSON snapshot as `:8000/health`, served alongside |

If `ADMIN_PANEL_PASSWORD` is unset the panel doesn't start at all.

---

## Local DevOps Simulation

Simulate the full canary deployment pipeline **without a Discord token or cloud cluster** using Docker Compose and synthetic traffic:

```bash
cd local-sim

make up                  # start stable stack (Grafana, Prometheus, nginx)
make deploy-canary       # introduce canary at 10% traffic
make watch-canary-auto   # auto promote or rollback based on metrics
make chaos-errors        # inject 25% error rate → triggers automatic rollback
make down                # tear everything down
```

Open Grafana at http://localhost:3000 (admin/admin) to see the **MisterFritz — Canary Deployment** dashboard with real-time error rates, P99 latency, and throughput split by version.

See [`local-sim/README.md`](local-sim/README.md) for the full guide including chaos experiments, canary controller options, and optional Kubernetes setup with kind.

### Production Kubernetes

For a real or local Kubernetes cluster, manifests are in `infra/k8s/`:

```bash
# Bootstrap a local kind cluster (installs Argo Rollouts, ArgoCD, Prometheus stack)
bash local-sim/kind/setup.sh

# Trigger a canary rollout
kubectl argo rollouts set image misterfritz misterfritz=<image> -n misterfritz
kubectl argo rollouts status misterfritz -n misterfritz --watch
```

The CI workflow (`.github/workflows/release.yml`) builds the image on tag push, runs a smoke test, and triggers a canary rollout with a manual approval gate.

---

## Troubleshooting

**"Ollama connection refused"**
Make sure Ollama is running: `ollama serve`. Check models with `ollama list`. Confirm `OLLAMA_HOST` matches the URL Ollama is listening on (`http://127.0.0.1:11434` for a local native install, `http://host.docker.internal:11434` when the bot runs in Docker on Mac/Windows, `http://172.17.0.1:11434` on Linux Docker). Quick connectivity check: `curl $OLLAMA_HOST/api/tags`.

**Ollama out-of-memory / model fails to load**
Symptoms: Ollama logs show `model requires more memory than available` or your machine swaps to disk. Use a smaller model — switch `THINKING_OLLAMA_MODEL` to `llama3.2` (or a 7B variant), or close other GPU consumers. With multiple models in play, Ollama keeps the most-recently-used one resident; set `OLLAMA_KEEP_ALIVE=0` to evict immediately after each request and trade latency for memory headroom.

**Very slow first response, fast subsequent responses**
This is normal — Ollama loads the model on the first request. If the wait is too long, the bot has a configurable hard timeout: set `OLLAMA_TIMEOUT=300` (seconds) in `.env` for very large models on CPU.

**Hung requests / bot stops responding**
Phase 1 added an Ollama request timeout (default 120s). If it trips, you'll see the agent return an error instead of hanging. Lower it (`OLLAMA_TIMEOUT=30`) to fail faster while you debug, or raise it for slow CPU inference. Check `:8000/health` for the bot's error rate and p99 latency snapshot.

**`execute_command` rejects a command**
The file-tools shell sandbox uses an allowlist (Phase 1). Allowed programs are listed in `EXEC_ALLOWED_COMMANDS` (see `.env.example`). To allow additional programs, override that env var. Shell features (pipes, `&&`, redirects) are not interpreted — the agent must run separate commands instead.

**Missing `DISCORD_BOT_TOKEN` error on startup**
Copy `.env.example` → `.env` and fill in the required values.

**Bot not responding in Discord**
Ensure "Message Content Intent" is enabled in the [Discord Developer Portal](https://discord.com/developers/applications) under your bot's settings.

**No documents found by `/lore`**
Add supported files (`.pdf`, `.docx`, `.xlsx`, `.csv`, `.txt`, `.md`) to the `input/` folder. The engine watches for changes and indexes automatically.

**Chroma DB "database is locked" or "could not acquire lock"**
Another process is holding the Chroma store — usually a previous bot instance that didn't exit cleanly. Stop all `python main_discord.py` processes, then restart. If it persists, delete `chroma_store/chroma.sqlite3-wal` and `chroma.sqlite3-shm` (the WAL files) — the main DB is safe.

**ffmpeg / ffprobe not found**
Set `FFMPEG_PATH` and `FFPROBE_PATH` in `.env` to the absolute paths of your installs, or put them on `PATH`. On Linux/macOS use `which ffmpeg` to find the path. On Windows the repo ships `ffmpeg.exe`/`ffprobe.exe` as a fallback for native runs.

**Image generation is slow or fails**
The first run downloads Stable Diffusion XL (~7 GB). GPU (CUDA) is strongly recommended. Check: `python -c "import torch; print(torch.cuda.is_available())"`.

**OCR not working for scanned PDFs**
Install optional deps: `pip install easyocr PyMuPDF pillow`.

**`/schedule add` says I have too many schedules**
There's a per-user cap (default 10) set by `MAX_SCHEDULES_PER_USER`. Use `/schedule list` to see yours, `/schedule remove <id>` to free a slot, or raise the cap in `.env`. Admins can view everyone's schedules with `/schedule list_all`.

---

## Contributing

Contributions are welcome. Please open an issue or pull request.

## License

See repository for license information.
