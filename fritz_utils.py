import json
import os
from enum import Enum

DOC_STORAGE_DESCRIPTION = """anything you don't know about."""
DOC_FOLDER = "./input"  # Folder containing your .docx and .pdf files
CHROMA_DB_PATH = "./chroma_store"  # Where the vector DB will be saved
CHROMA_COLLECTION_NAME = "word_docs_rag"
CHAT_DB_NAME = "chat_history.db"
INDEXED_FILES_PATH = os.path.join(CHROMA_DB_PATH, "indexed_files.txt")
THINKING_OLLAMA_MODEL = "gpt-oss"
FAST_OLLAMA_MODEL = "llama3.2"

class MessageSource(Enum):
    DISCORD_TEXT = 0,
    DISCORD_VOICE = 1,
    LOCAL = 2

def get_key_from_json_config_file(key_name: str) -> str | None:
    file_path = "config.json"
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            return data.get(key_name)  # Get the key value by key name
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
    except json.JSONDecodeError:
        print(f"Error: The file at {file_path} is not a valid JSON file.")
    except Exception as e:
        print(f"Error reading file: {e}")
    return None

dcd = get_key_from_json_config_file("doc_storage_description")
if dcd:
    DOC_STORAGE_DESCRIPTION = get_key_from_json_config_file("doc_storage_description")