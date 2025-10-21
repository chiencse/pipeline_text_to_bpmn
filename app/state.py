import json
import os
from datetime import datetime

from typing import Dict, Any, List
from typing_extensions import TypedDict

class PipelineState(TypedDict, total=False):
    text: str
    chunks: List[str]
    entities: List[Dict]
    relations: List[Dict]
    candidates: List[Dict]
    bpmn: Dict[str, Any]
    mapping: List[Dict]
    config: Dict[str, Any]
    meta: Dict[str, Any]

def log_state(state, step=None, log_dir="./app/viz/logs"):
    """
    Log the current state to a file for debugging/visualization.
    Each log is timestamped and optionally includes the step name.
    """
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    fname = f"state_{ts}"
    if step:
        fname += f"_{step}"
    fname += ".json"
    path = os.path.join(log_dir, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    return path
