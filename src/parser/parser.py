import json
from typing import List
from src.models.models import FunctionDefinition, PromptInput


def load_functions(path: str) -> List[FunctionDefinition]:
    """
    Load and validate function definitions from JSON file.
    """
    functions = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            try:
                functions.append(FunctionDefinition(**item))
            except Exception as e:
                print(f"[WARN] Skipping invalid function entry: {item} ({e})")
        return functions

    except FileNotFoundError:
        print(f"[ERROR] File not found: {path}")
        return []

    except json.JSONDecodeError:
        print(f"[ERROR] Invalid JSON in file: {path}")
        return []

    except Exception as e:
        print(f"[ERROR] Unexpected error loading functions: {e}")
        return []


def load_prompts(path: str) -> List[PromptInput]:
    """
    Load and validate prompts from JSON file.
    """
    prompts = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            try:
                if isinstance(item, str):
                    prompts.append(PromptInput(prompt=item))
                elif isinstance(item, dict) and "prompt" in item:
                    prompts.append(PromptInput(**item))
            except Exception as e:
                print(f"[WARN] Skipping invalid prompt entry: {item} ({e})")
        return prompts

    except FileNotFoundError:
        print(f"[ERROR] File not found: {path}")
        return []

    except json.JSONDecodeError:
        print(f"[ERROR] Invalid JSON in file: {path}")
        return []

    except Exception as e:
        print(f"[ERROR] Unexpected error loading prompts: {e}")
        return []
