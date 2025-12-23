# Mister Fritz - Discord AI Chatbot

Mister Fritz is an AI-powered Discord bot with a sophisticated, sardonic personality modeled after an English butler. The bot uses LangChain, LangGraph, and Ollama to provide conversational responses with memory retention, web search capabilities, and document retrieval.

## Features

- **Conversational AI**: Witty, butler-like responses powered by local LLM models via Ollama
- **Memory System**: Stores and retrieves conversation summaries per user using ChromaDB
- **Document Search**: RAG (Retrieval-Augmented Generation) system for querying local documents (Word docs & PDFs)
- **Web Integration**: Can search the web and scrape websites for information
- **Discord Commands**:
  - `$hello` - Greet the bot
  - `$lore <query>` - Search local documents
  - `$join` / `$leave` - Voice channel management
  - Direct messages or mentions trigger conversational responses
- **Tools**: Dice rolling, current time lookup, web search, document search, and memory retrieval

## Architecture

The bot uses a LangGraph-based agent system with:
- **Conversation Node**: Handles user queries with tool access
- **Summarization Node**: Automatically summarizes long conversations and stores memories
- **SQLite Checkpointing**: Persists conversation state
- **ChromaDB**: Stores user memories and document embeddings

## Prerequisites

### 1. Install Python
- Python 3.12+ is recommended
- Download from [python.org](https://www.python.org/downloads/)

### 2. Install Ollama

**Windows:**
1. Download the Ollama installer from [ollama.com](https://ollama.com/download/windows)
2. Run the installer and follow the prompts
3. Ollama will run as a background service

**macOS:**
```bash
# Using Homebrew
brew install ollama

# Or download from ollama.com/download/mac
```

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 3. Pull Required Ollama Models

After installing Ollama, pull the models used by Mister Fritz:

```bash
# Main conversation model (custom model - see modelfiles/ directory)
ollama pull gpt-oss

# Fast model for quick operations
ollama pull llama3.2

# Embedding model for document search
ollama pull mxbai-embed-large
```

**Note:** The `gpt-oss` model appears to be a custom modelfile. Check the `modelfiles/` directory for the model definition and create it with:
```bash
ollama create gpt-oss -f modelfiles/gpt-oss.modelfile
```

## Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd MisterFritz
```

2. **Create a virtual environment:**
```bash
python -m venv .venv
```

3. **Activate the virtual environment:**

Windows:
```bash
.venv\Scripts\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

4. **Install dependencies:**
```bash
pip install discord.py langchain langchain-ollama langchain-chroma langgraph duckduckgo-search beautifulsoup4 requests pytz chromadb unstructured python-docx pypdf pillow easyocr PyMuPDF
```

5. **Configure the bot:**

Create a `config.json` file in the project root with the following structure:
```json
{
  "discord_bot_token": "YOUR_DISCORD_BOT_TOKEN",
  "doc_storage_description": "Description of what documents contain (e.g., 'The world of Sennen or Dungeons and Dragons')",
  "root_user_id": "your_username"
}
```

To get a Discord bot token:
1. Go to [Discord Developer Portal](https://discord.com/developers/applications)
2. Create a new application
3. Go to the "Bot" section
4. Click "Reset Token" to generate a token
5. Enable "Message Content Intent" under Privileged Gateway Intents
6. Invite the bot to your server using the OAuth2 URL generator

6. **Set up document folder (optional):**

Create an `input/` directory and add `.docx` or `.pdf` files for the RAG system:
```bash
mkdir input
# Add your documents to this folder
```

## Running the Bot

1. **Start Ollama** (if not already running):
```bash
ollama serve
```

2. **Run the Discord bot:**
```bash
python main_discord.py
```

The bot will:
- Connect to Discord
- Initialize the vector store for document search
- Begin responding to messages and commands

## Usage Examples

**Direct conversation:**
```
@Botname What's the weather like today?
```

**Search documents:**
```
$lore Tell me about the kingdom of Sennen
```

**Roll dice:**
```
@Botname roll 2d20
```

**Web search:**
```
@Botname What's the latest news about AI?
```

## Troubleshooting

**"Ollama connection refused":**
- Ensure Ollama is running with `ollama serve`
- Check if models are installed with `ollama list`

**"No documents found":**
- Verify documents are in the `input/` folder
- Supported formats: `.docx`, `.pdf`

**Discord bot not responding:**
- Verify the bot token in `config.json`
- Ensure "Message Content Intent" is enabled in Discord Developer Portal
- Check bot permissions in your Discord server

**OCR not working for PDFs:**
- Install optional dependencies: `pip install easyocr PyMuPDF pillow`

## License

See repository for license information.

## Contributing

Contributions are welcome! Please open an issue or pull request.
