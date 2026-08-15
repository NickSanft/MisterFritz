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

Only needed if you install the `image` or `voice` extras. Install torch *first*,
from the index matching your hardware, so pip does not pull generic CUDA wheels:

```bash
# CUDA (Windows/Linux)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU-only / macOS
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
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

That installs the pinned production set. **The core is deliberately torch-free** —
Discord, the agent, memory, document RAG and the admin/chat panel, with none of
the multi-GB GPU stack. Optional features live in extras:

| Extra | Enables | Pulls torch |
|---|---|---|
| `voice` | `/voice` text-to-speech and Whisper transcription | yes |
| `image` | `/gen` image generation (SDXL) | yes |
| `ocr` | OCR fallback for scanned PDFs | yes |
| `telegram` | `main_telegram.py` | no |
| `dev` | pytest, coverage, ruff | no |

```bash
pip install ".[voice]"          # one extra
pip install ".[voice,image]"    # several
pip install ".[all]"            # everything
```

Without an extra its feature reports itself unavailable rather than crashing:
`/voice` replies that it has no voice, and scanned-PDF OCR is skipped.

> Dependencies are declared in `pyproject.toml`'s `[project]` table;
> `requirements.txt` is the pinned lock of that set. Add new dependencies to
> `pyproject.toml` first. Do **not** regenerate the lock with
> `pip freeze > requirements.txt` — that is how a 2016-era neuroimaging stack
> got in, and `tests/test_packaging.py` will fail if it comes back.

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
# Your numeric Discord ID, not your username — see the Identity section.
ROOT_USER=discord-123456789012345678
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
| `ROOT_USER` | — | **Required.** Canonical identity with admin privileges (`/workspace set`, `/schedule add/remove`). Use `discord-<your numeric id>`, not your username — see [Identity](#identity). |
| `ADMIN_USERS` | — | Comma-separated additional admin identities, same canonical form. Anyone listed gets the same powers as `ROOT_USER`. |
| `THREADS_PER_CHANNEL` | `false` | One conversation thread per channel instead of one per person. Turning it on branches every existing conversation — see [Identity](#identity). |
| `ADMIN_LEGACY_NAME_MATCH` | `false` | Also match `ROOT_USER`/`ADMIN_USERS` against display names. Compatibility shim only; with it on, taking your username takes your admin rights. |
| `IDENTITY_LINKS` | — | `web-alice=discord-123,…` — treat one identity as another, so memories and conversation follow you across surfaces. |
| `WORKSPACES_ROOT` | `./workspaces` | Parent directory for per-user sandboxed workspaces created via `/workspace enable`. |
| `AUDIT_LOG_PATH` | `audit.log` | Path to the append-only NDJSON audit log. Captures `/forget`, `/export`, admin-panel mutations, and every file-tool write / edit / shell-exec. |
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

## Identity

Every store — memories, conversation threads, schedules, workspaces, the admin gate — is keyed off a **canonical identity**:

```
<platform>-<immutable id>      discord-123456789   telegram-987654321   web-alice
```

A dash, never a colon: a colon is illegal in a Windows filename and silently creates an NTFS alternate data stream on write, and identities reach filenames in several places.

The id half is the platform's *immutable* id wherever one exists — a Discord or Telegram snowflake, not a display name. That is the whole point:

- Renaming your Discord account no longer orphans your memories, workspace, schedules and admin rights.
- A Telegram user and a Discord user who happen to share a handle no longer share a keyspace.
- `/forget memories` actually deletes. It used to delete the namespace for the raw name while the write path used a stripped one, so for any username containing punctuation it reported success and removed nothing.

Fritz still addresses you by name — a `user_aliases` table carries the display name alongside the id, and it's what the scheduler uses when a cron job fires hours later with no live user object.

### Migrating an existing install

Keys change in five stores, so run the migration **with the bot stopped**, before the first start on this version. Ollama does not need to be running.

```bash
python migrate_identity.py --dry-run
```

Dry run is the default — it reports every distinct key it finds in each store and flags any without a mapping. Note the canonical id for each, then:

```bash
python migrate_identity.py --map divora=discord-123456789 --apply
```

`--apply` writes `identity_migration_<timestamp>.json` *before* touching anything; pass it back via `--reverse` to undo the run exactly. A second `--apply` is a no-op. It refuses to run if any discovered key is unmapped, rather than leaving the stores half-migrated.

Then set `ROOT_USER` to the canonical form in `.env` — **before restarting, or you lose admin commands**. Get your numeric Discord id by enabling Developer Mode and right-clicking your name → Copy User ID.

### Per-channel threads

`THREADS_PER_CHANNEL=true` gives each identity one thread per channel (`discord-123#456`) instead of one thread everywhere, so a conversation in #general stops bleeding into your DMs.

Off by default, because **turning it on branches every existing conversation**: the identity-only thread stays in the database untouched, but new messages start a fresh per-channel thread and the old context is no longer read. Flip it once, deliberately.

`/forget conversation` sweeps the identity thread and every channel thread under it, and cannot catch a sibling that shares a prefix — `discord-1` does not touch `discord-10`.

---

## Architecture

```
User (Discord)
    │
    ▼
main_discord.py   ──  slash commands, on_message, streaming handler
    │
    ▼
mister_fritz.py   ──  LangGraph state machine: START → executor → (summarize | END)
    ├─ EXECUTOR   ──  ReAct agent, tool dispatch, token streaming
    │   ├─ get_current_time, roll_dice
    │   ├─ scrape_web, search_web
    │   ├─ search_documents  →  document_engine  →  Chroma RAG
    │   ├─ search_memories   →  chroma_store     →  Chroma KV
    │   ├─ generate_image    →  image_generator  →  SDXL
    │   ├─ analyze_image     →  Ollama LLaVA
    │   └─ file tools (admin only) →  file_tools.py
    └─ SUMMARIZE  ──  trims at 30+ messages; the summary itself is written to
                      Chroma on a background thread, off the critical path
```

Conversation state is checkpointed per-user in `chat_history.db` (SQLite). Each
turn the executor replays the newest slice of that transcript that fits
`HISTORY_TOKEN_BUDGET` (default 4096 tokens) into the ReAct agent — that is the
short-term memory, and it is the only one the sub-agent has, since it is
compiled without a checkpointer of its own. Anything older is reachable only
through the Chroma memory store, which the summariser writes to and which is
auto-injected into the system prompt (capped at `MEMORY_INJECT_MAX_CHARS`) —
long-term recall. Set `HISTORY_TOKEN_BUDGET=0` to disable the window and send
only the latest message.

---

## Testing

```bash
pip install -e ".[dev]"
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
- `mister_fritz` — history window trimming, executor inputs, token-delta streaming, off-path summarisation, memory-key slugs
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

Mutating actions (POST-only, JS-confirmed in the UI):
- **Forget everything about a user** (user detail page) — runs the same op as `/forget all`.
- **Disable workspace** (user detail page) — drops the registration; files on disk are kept.
- **Cancel** (next to each row on `/schedules`) — bypasses ownership check since an admin is acting.
- **Re-index** (next to each row on `/documents`) — re-enqueues the file for ingestion.

Every mutating action writes a line to `AUDIT_LOG_PATH` with the Basic-auth username, the target, and the result.

If `ADMIN_PANEL_PASSWORD` is unset the panel doesn't start at all.

---

## Chat UI

Talk to Fritz from a browser at **`http://127.0.0.1:8001/chat`** — no Discord required. It runs on the same server as the admin panel, but with its own identity model: a shared password plus a cookie-carried username.

### Getting in

Open `/chat`, enter `CHAT_PASSWORD` and pick a username, start chatting.

Your web identity is **separate from your Discord one**: signing in as `alice` gives you `web-alice`, which is a different key from `discord-123456789` no matter what name that account happens to use. Earlier versions shared one thread across both surfaces by matching on the username — that is exactly how a chat session could read and overwrite the Discord conversation of whoever owned that name.

To deliberately join them, link the identities:

```
IDENTITY_LINKS=web-alice=discord-123456789
```

That folds the *whole* identity — memories, schedules, workspace and conversation — not just the thread, so a topic you start in Discord can genuinely be continued in the browser.

> **Trust model.** `CHAT_PASSWORD` is the perimeter ("may you be here at all"); the username is only namespacing ("whose memories and conversation"). Anyone holding the password can still claim any name and read that person's chat history — set `CHAT_ALLOWED_USERS` to narrow it. It's meant for "you and your friends on a port-forwarded local network," not the public internet. If you need real per-user auth, keep the port bound to localhost (the default) and tunnel over SSH.

The identity cookie is HMAC-signed (so it can't be tampered to impersonate another user mid-session), `httponly`, `SameSite=Lax`, and rolls forward a 30-day expiry on every visit. The signing secret comes from `CHAT_COOKIE_SECRET`; if unset, one is generated and saved to `.chat_cookie_secret` (gitignored) on first boot.

### What it does

| Feature | Notes |
|---|---|
| **Streaming responses** | Fritz's reply appears token-by-token via Server-Sent Events (delta frames), with a blinking cursor while he writes. A `reset` frame clears the bubble when he starts a fresh answer — e.g. after narrating, calling a tool, and beginning again. |
| **Markdown rendering** | Code blocks, tables, lists, bold/italic all render. Markdown is finalised when the response completes. |
| **Tool progress** | Ephemeral italic lines ("🔍 Searching the web…", "🧠 Looking through my memories…") show what Fritz is doing, then vanish when the answer lands. |
| **Conversation history** | The page hydrates with your last 40 messages on load, so a refresh doesn't lose context. |
| **New conversation** | A header button resets just this thread's context (calls `forget_conversation`). Your memories and schedules are untouched. |
| **Image analysis** | Drag an image anywhere onto the chat, or click 📎, then send a message — Fritz analyses it with the vision model. |
| **Inline images** | Images Fritz generates render directly in his reply bubble. |
| **Document upload** *(admin only)* | Admins get a "Add to shared docs" control that drops a file into `DOC_FOLDER`; the watchdog auto-indexes it for `/lore` and `search_documents`. |

No JavaScript? The message form still works — it falls back to a synchronous submit-and-render (you just don't get streaming).

### Configuration

| Variable | Default | Description |
|---|---|---|
| `CHAT_COOKIE_SECRET` | auto-generated | HMAC secret for the identity cookie. Set this explicitly if you run multiple instances or want stable cookies across redeploys. |
| `CHAT_IMAGE_UPLOAD_MAX_BYTES` | `10485760` (10 MB) | Max size for an uploaded image. |
| `CHAT_DOC_UPLOAD_MAX_BYTES` | `10485760` (10 MB) | Max size for an admin document upload. |

Image uploads accept JPEG / PNG / WEBP / GIF only. Document uploads accept the same extensions as the RAG engine (`.pdf`, `.docx`, `.xlsx`, `.csv`, `.txt`, `.md`). Every chat turn and every upload is written to `AUDIT_LOG_PATH` (`chat_message`, `chat_upload_image`, `chat_upload_document`, `chat_login`/`chat_logout` events).

### Remote access

Bound to `127.0.0.1` like the admin panel. To reach it from another machine, SSH-forward the port rather than exposing it:

```bash
ssh -L 8001:127.0.0.1:8001 you@your-bot-host
# then open http://127.0.0.1:8001/chat locally
```

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
The bot now pre-warms every configured model in a background thread at startup, so the first DM shouldn't pay the cold-load tax anymore. If you still see latency on the first response, Ollama is unloading models between requests — increase `OLLAMA_KEEP_ALIVE` in `.env` (default `5m`, set to `-1` to pin models forever at the cost of permanent VRAM use). For very large models on CPU, also raise `OLLAMA_TIMEOUT=300`.

**Hung requests / bot stops responding**
Phase 1 added an Ollama request timeout (default 120s). If it trips, you'll see the agent return an error instead of hanging. Lower it (`OLLAMA_TIMEOUT=30`) to fail faster while you debug, or raise it for slow CPU inference. Check `:8000/health` for the bot's error rate and p99 latency snapshot.

**`execute_command` rejects a command**
The file-tools shell sandbox uses an allowlist. Allowed programs are listed in `EXEC_ALLOWED_COMMANDS` (see `.env.example`). Three other rules can reject a command: the program name must be bare (`python`, not `./python` or `C:\...\python.exe`); arguments may not contain `..` or absolute paths outside the workspace; and while `EXEC_REQUIRE_ADMIN=true` (the default) only admins may run programs at all — the other five file tools stay open to every workspace holder. Shell features (pipes, `&&`, redirects) are not interpreted — the agent must run separate commands instead. Commands run with a scrubbed environment: only `PATH` plus a few platform basics (`EXEC_ENV_PASSTHROUGH`) are passed through, so a script cannot read the bot's tokens out of `os.environ`. If a command works from a terminal but fails through the bot with a "not found" or TLS error, the fix is to add the variable it needs to `EXEC_ENV_PASSTHROUGH`.

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
Install the OCR extra: `pip install ".[ocr]"`.

If `import fitz` fails or `fitz.open` is missing, a distribution named `fitz`
has been installed alongside PyMuPDF — it is unrelated neuroimaging software
that claims the same `fitz/` directory. Fix it with:

```bash
pip uninstall -y fitz && pip install --force-reinstall PyMuPDF
```

The force-reinstall is not optional: `fitz`'s file manifest lists
`fitz/__init__.py`, so uninstalling it deletes PyMuPDF's shim on the way out.

**`/schedule add` says I have too many schedules**
There's a per-user cap (default 10) set by `MAX_SCHEDULES_PER_USER`. Use `/schedule list` to see yours, `/schedule remove <id>` to free a slot, or raise the cap in `.env`. Admins can view everyone's schedules with `/schedule list_all`.

---

## Contributing

Contributions are welcome. Please open an issue or pull request.

## License

See repository for license information.
