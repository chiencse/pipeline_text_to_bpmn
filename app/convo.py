# app/convo.py
from typing import Dict, Any, List
from app.state import ChatMessage, Feedback

def add_msg(state, role: str, content: str, **meta):
    msgs: List[ChatMessage] = state.get("chat", []) or []
    msgs.append({"role": role, "content": content, "meta": meta})
    state["chat"] = msgs
    return state

def add_feedback(state, fb: Feedback):
    q = state.get("feedback", []) or []
    q.append(fb)
    state["feedback"] = q
    return state
