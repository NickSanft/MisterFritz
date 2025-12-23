import json
from enum import Enum

DOCUMENT_STORAGE_DESCRIPTION = """Stuff"""

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