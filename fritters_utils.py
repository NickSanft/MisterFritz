import json

DISCORD_KEY = "discord_bot_token"
ROOT_USER_ID_KEY = "root_user_id"

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

def has_key_from_json_config_file(key: str) -> bool:
    if get_key_from_json_config_file(key) is None:
        return False
    return True

def check_root_user(user_id: str) -> bool:
    root_user_id = get_key_from_json_config_file(ROOT_USER_ID_KEY)
    if root_user_id is None or user_id != root_user_id:
        return False
    return True