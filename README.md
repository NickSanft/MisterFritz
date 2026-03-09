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
| Observability | Live health metrics via `/health`, with Prometheus + Grafana support |

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
| `/workspace <path>` | (Root user only) Set the file operations workspace |

Direct messages or `@mentions` trigger the full agent with all tools.
Attach images or Discord voice messages — Fritz can analyze both.

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

**Configure the bot** — copy `.env.example` to `.env` and fill in your values:
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
| `ROOT_USER` | — | **Required.** Discord username with file operation access |
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
- `file_tools` — all 6 tools, authorization, path traversal prevention
- `observability` — counters, latency rolling window, error tracking, health text
- `agent_tools` — `get_current_time`, `format_prompt`, `scrape_web` (mocked), `search_web` (mocked), `roll_dice`
- `discord_commands` — `split_into_chunks`, `StreamingMessageHandler` rate limiting

CI runs on every push via GitHub Actions (see `.github/workflows/ci.yml`).

---

## Troubleshooting

**"Ollama connection refused"**
Make sure Ollama is running: `ollama serve`. Check models with `ollama list`.

**Missing `DISCORD_BOT_TOKEN` error on startup**
Copy `.env.example` → `.env` and fill in the required values.

**Bot not responding in Discord**
Ensure "Message Content Intent" is enabled in the [Discord Developer Portal](https://discord.com/developers/applications) under your bot's settings.

**No documents found by `/lore`**
Add supported files (`.pdf`, `.docx`, `.xlsx`, `.csv`, `.txt`, `.md`) to the `input/` folder. The engine watches for changes and indexes automatically.

**Image generation is slow or fails**
The first run downloads Stable Diffusion XL (~7 GB). GPU (CUDA) is strongly recommended. Check: `python -c "import torch; print(torch.cuda.is_available())"`.

**OCR not working for scanned PDFs**
Install optional deps: `pip install easyocr PyMuPDF pillow`.

---

## Contributing

Contributions are welcome. Please open an issue or pull request.

## License

See repository for license information.
