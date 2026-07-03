"""
macro_store.py — Persist user-created custom macros across sessions.

Only *custom* macros (those a user builds at runtime) are saved; the built-in
defaults live in config.py. The store is a small JSON file next to the app.
"""

import json
import os

import config

_STORE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           config.USER_MACRO_STORE)


def load_into_config() -> int:
    """Merge persisted custom macros into config.MACRO_DEFINITIONS.

    Returns the number of custom macros loaded.
    """
    if not os.path.exists(_STORE_PATH):
        return 0
    try:
        with open(_STORE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return 0

    macros = data.get("macros", {})
    count = 0
    for macro_id, defn in macros.items():
        defn["custom"] = True
        config.MACRO_DEFINITIONS[macro_id] = defn
        count += 1
    return count


def save_from_config() -> bool:
    """Write every custom macro currently in config.MACRO_DEFINITIONS to disk."""
    customs = {mid: defn for mid, defn in config.MACRO_DEFINITIONS.items()
               if defn.get("custom")}
    try:
        with open(_STORE_PATH, "w", encoding="utf-8") as f:
            json.dump({"macros": customs}, f, indent=2)
        return True
    except OSError:
        return False


def add_macro(macro_id: str, definition: dict) -> bool:
    """Register a new custom macro and persist it."""
    definition = dict(definition)
    definition["custom"] = True
    config.MACRO_DEFINITIONS[macro_id] = definition
    return save_from_config()


def delete_macro(macro_id: str) -> bool:
    """Remove a custom macro (built-ins are protected) and persist."""
    defn = config.MACRO_DEFINITIONS.get(macro_id)
    if not defn or not defn.get("custom"):
        return False
    config.MACRO_DEFINITIONS.pop(macro_id, None)
    return save_from_config()


def store_path() -> str:
    return _STORE_PATH
