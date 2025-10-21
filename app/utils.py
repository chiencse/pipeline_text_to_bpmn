import json, os, re
from typing import List, Dict, Tuple

STORAGE_DIR = os.path.join(os.path.dirname(__file__), "storage")
os.makedirs(STORAGE_DIR, exist_ok=True)
MEMORY_FILE = os.path.join(STORAGE_DIR, "memory_kv.json")
FEEDBACK_FILE = os.path.join(STORAGE_DIR, "feedback.json")

def load_json(path: str, default):
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(default, f, ensure_ascii=False, indent=2)
        return default
    return json.load(open(path, "r", encoding="utf-8"))

def save_json(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def memory_get(key: str, default=None):
    kv = load_json(MEMORY_FILE, {})
    return kv.get(key, default)

def memory_set(key: str, value):
    kv = load_json(MEMORY_FILE, {})
    kv[key] = value
    save_json(MEMORY_FILE, kv)

def add_feedback(record: Dict):
    arr = load_json(FEEDBACK_FILE, [])
    arr.append(record)
    save_json(FEEDBACK_FILE, arr)

def simple_chunk(text: str, max_len: int = 1200) -> List[str]:
    text = text.strip()
    if len(text) <= max_len:
        return [text]
    # naive split by sentence and group
    sents = re.split(r'(?<=[.!?])\s+', text)
    chunks, cur = [], ""
    for s in sents:
        if len(cur) + len(s) + 1 <= max_len:
            cur += (" " if cur else "") + s
        else:
            if cur: chunks.append(cur)
            cur = s
    if cur: chunks.append(cur)
    return chunks
