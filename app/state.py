import json
import os
from datetime import datetime

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
class CheckpointModel(BaseModel):
    step_idx: int
    timestamp: datetime
    snapshot: Dict[str, Any] = Field(default_factory=dict)
    note: Optional[str] = None
class PipelineState(TypedDict, total=False):
    thread_id: str
    text: str
    chunks: List[str]
    preprocess: Dict[str, Any]
    entities: List[Dict]
    relations: List[Dict]
    candidates: List[Dict]
    bpmn: Dict[str, Any]
    mapping: List[Dict]
    config: Dict[str, Any]
    meta: Dict[str, Any]
    syntax: List[Dict]
    loop_counters: Dict[str, int] = Field(default_factory=dict)
    checkpoints: List[CheckpointModel] = Field(default_factory=list)
    user_feedback: List[Dict]
    user_decision: Optional[str]
    user_updated_text: Optional[str]
    user_feedback_text: Optional[str]  # Feedback text for BPMN rejection
    user_mapping_decision: Optional[str]  # Decision for mapping feedback
    user_mapping_feedback_text: Optional[str]  # Feedback text for mapping rejection
    selected_node_ids: Optional[List[str]]  # Selected node IDs for mapping feedback
    retrieval_feedback_text: Optional[str]  # Feedback text to incorporate in retrieval
    # Interrupt handling fields
    __pending_interrupt__: Optional[Dict]  # Stores interrupt payload for FastAPI
    __interrupt_node__: Optional[str]  # Stores node name that triggered interrupt
    # Render output fields
    render_xml: Optional[str]           # BPMN XML for visualization
    render_activities: List[Dict]  # Activities for RPA
    render_variables: List[Dict]   # Variables (placeholder)

def log_state(state, step=None, log_dir="./app/viz/logs", use_db=False):
    """
    Log the current state to a file for debugging/visualization.
    Optionally also log to PostgreSQL database.
    
    Args:
        state: Pipeline state dictionary
        step: Optional step name (e.g., "pipeline_a", "pipeline_b")
        log_dir: Directory for file logs (default: "./app/viz/logs")
        use_db: If True, also log to PostgreSQL database (default: False)
    
    Returns:
        path: Path to the log file
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
    
    # Optionally log to PostgreSQL
    if use_db:
        try:
            from app.logs.db_logger import log_state_to_db
            log_state_to_db(state, step=step)
        except Exception as e:
            # Don't fail if DB logging fails, just log the error
            import logging
            logging.warning(f"Failed to log to database: {e}")
    
    return path