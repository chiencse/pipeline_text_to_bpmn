import math
from typing import Any, Dict, List
from langgraph.graph import StateGraph, END
import numpy as np
from pydantic import BaseModel
from app.pos_dep import node_pos_dep
from app.preprocessing import node_preprocess
from app.state import PipelineState
from app.utils import simple_chunk
from app.retrieval import _normalize, hybrid_search
from app.validator import validate_bpmn
from app.llm import call_llm_ner, call_llm_bpmn_with_mapping
from app.bpmn_converter import render_bpmn_output
from langgraph.types import Command, interrupt

def _count_distinct_pkgs(entities: List[Dict[str, Any]]) -> int:
    pkgs = [e.get("pkg") for e in entities if e.get("pkg")]
    return len(set(pkgs))

def compute_dynamic_k(
    query: str,
    entities: List[Dict[str, Any]] | None = None,
    *,
    base_k: int = 5,
    min_k: int = 5,
    max_k: int = 30,
) -> int:
    """
    Heuristic tính số lượng kết quả cần lấy (k) cho retrieval.
    Ảnh hưởng bởi:
      - Độ dài query (số từ)
      - Số entity (nếu có)
      - Số distinct package (nếu có)
    """

    entities = entities or []
    n_entities = len(entities)
    n_pkgs = _count_distinct_pkgs(entities) if callable(globals().get("_count_distinct_pkgs")) else 0

    # --- Bắt đầu với base
    k = base_k

    # --- Yếu tố entity
    if n_entities:
        k += int(math.ceil(1.5 * n_entities))
    if n_pkgs > 1:
        k += 2 * (n_pkgs - 1)

    # --- Yếu tố độ dài query
    n_words = len((query or "").split())
    if n_words > 0:
        # với query dài, tăng nhẹ k; query ngắn, giảm nhẹ
        if n_words <= 6:
            k -= 1
        elif n_words <= 12:
            k += 1
        elif n_words <= 25:
            k += 2
        else:
            k += 3 + n_words // 20  # thêm nếu quá dài (đoạn mô tả dài)

    # --- Clamp trong khoảng
    k = max(min_k, min(k, max_k))

    return int(k)
# def node_preprocess(state: PipelineState):
#     state["chunks"] = simple_chunk(state["text"])
#     return state

def node_ner_llm(state: PipelineState):
    state["entities"] = call_llm_ner(state)
    return state

def node_retrieve(state: "PipelineState"):
    """
    Per-sentence retrieval:
      - iterate sentences in state['chunks'][].sentences[]
      - use sentence.text as query (no entities)
      - call hybrid_search per sentence
      - merge candidates by activity_id (sum scores), normalize, return top final_k
      ? If sentence has some word as email, gmail, SAP, Odoo, ErpNext. Should we add certainly package to candidate?
    """
    chunks = state.get("chunks", []) or []
    if not chunks:
        state["candidates"] = []
        state["debug_retrieve"] = {"reason": "no_chunks"}
        return state

    per_sent_queries = {}
    per_sent_results = {}
    merged = {}
    sent_idx = 0
    default_k_per_sent = 5

    for chunk in chunks:
        for sent in chunk.get("sentences", []) or []:
            q = (sent.get("text") or "").strip()
            if not q:
                sent_idx += 1
                continue

            sid = sent_idx
            sent_idx += 1
            per_sent_queries[sid] = q

            try:
                dyn_k = compute_dynamic_k(q, [], base_k=5, min_k=3, max_k=20)
            except Exception:
                dyn_k = default_k_per_sent

            hits = hybrid_search(q, k=dyn_k) or []
            per_sent_results[sid] = hits

            for h in hits:
                tid = h.get("activity_id") or h.get("templateId") or h.get("pkg") or h.get("keyword")
                if not tid:
                    continue
                score = float(h.get("score", 0.0))
                if tid not in merged:
                    merged[tid] = {
                        "activity_id": h.get("activity_id") or h.get("templateId"),
                        "pkg": h.get("pkg"),
                        "keyword": h.get("keyword"),
                        "text": h.get("text"),
                        "score_sum": score,
                        "count": 1,
                        "sources": {sid: score}
                    }
                else:
                    merged[tid]["score_sum"] += score
                    merged[tid]["count"] += 1
                    merged[tid]["sources"][sid] = score

    if not merged:
        state["candidates"] = []
        state["debug_retrieve"] = {"per_sent_queries": per_sent_queries, "merged_count": 0}
        return state

    ids = list(merged.keys())
    sums = np.array([merged[i]["score_sum"] for i in ids], dtype=float)
    norm = _normalize(sums)

    out = []
    for i, tid in enumerate(ids):
        v = merged[tid]
        out.append({
            "activity_id": v.get("activity_id"),
            "pkg": v.get("pkg"),
            "keyword": v.get("keyword"),
            "text": v.get("text"),
            "score": float(norm[i]),
            "score_sum": float(v.get("score_sum")),
            "count": int(v.get("count")),
            "sources": v.get("sources"),
        })

    out.sort(key=lambda x: x["score"], reverse=True)

    try:
        final_k = compute_dynamic_k(" ".join(per_sent_queries.values()), [], base_k=5, min_k=5, max_k=30)
    except Exception:
        final_k = 10
    final_k = min(final_k, 30)

    state["candidates"] = out[:final_k]
    state["candidates"].append({
        "debug_retrieve": {
            "n_sentences": sent_idx,
            "per_sent_queries": per_sent_queries,
            "merged_count": len(out),
            "final_k": final_k,
        }
    })
    return state


def node_bpmn_with_map(state: PipelineState):
    out = call_llm_bpmn_with_mapping(
        text=state["text"], entities=state["entities"], candidates=state["candidates"], syntax_parser=state.get("syntax", {})
    )
    state["bpmn"] = out.bpmn
    state["mapping"] = [m.dict() for m in (out.mapping or [])]
    state["entities"] = [e.dict() for e in out.entities]
    state["relations"] = [r.dict() for r in out.relations]
    return state

def node_validate(state: PipelineState):
    errors = validate_bpmn(state["bpmn"], scenario="A")
    state["meta"] = {"errors": errors}
    return state

def node_render(state: PipelineState):
    """
    Render node: Generate BPMN XML, activities, and variables from bpmn/mapping data.
    Uses the bpmn_converter module (ported from json-to-bpmn-xml.util.ts).
    
    Output fields:
    - render_xml: BPMN 2.0 XML string for visualization
    - render_activities: List of activities with properties for RPA system
    - render_variables: List of variables (placeholder for future)
    - render_snapshot: Legacy snapshot for compatibility
    """
    bpmn = state.get("bpmn") or {}
    mapping = state.get("mapping") or []
    
    # Call the converter to generate XML, activities, and variables
    result = render_bpmn_output(bpmn, mapping)
    
    if result.get("success"):
        state["render_xml"] = result.get("xml")
        state["render_activities"] = result.get("activities") or []
        state["render_variables"] = result.get("variables") or []
    else:
        # Store errors in meta if conversion failed
        errors = result.get("errors") or []
        state.setdefault("meta", {})["render_errors"] = errors
        state["render_xml"] = None
        state["render_activities"] = []
        state["render_variables"] = []
    
    # Keep legacy render_snapshot for backward compatibility
    state["render_snapshot"] = {
        "bpmn": bpmn,
        "mapping": mapping,
        "meta": state.get("meta"),
        "xml": state.get("render_xml"),
        "activities": state.get("render_activities"),
        "variables": state.get("render_variables"),
    }
    
    return state

def apply_router(state: PipelineState):
    v = state["user_decision"]
    print("=== APPLY ROUTER ===" , v)
    return "update" if v == "update" else "approve"
def node_user_feedback(state: PipelineState):
    print("=== USER FEEDBACK NODE ===")
    # Nếu user_decision chưa có -> pause and surface payload to FE
    if "user_decision" not in state or not state.get("user_decision"):
        payload = {
            "instruction": "Do you approve this action? Reply JSON: {'user_decision': 'approve'} or {'user_decision':'update','user_updated_text':'...'}",
            "draft_snapshot": state.get("draft_snapshot", state.get("text", "Không có dữ liệu")),
            "hint": "Approve or Update"
        }
        res = interrupt(payload)   
        state["user_decision"] = res.get("user_decision")
        state["user_updated_text"] = res.get("user_updated_text")

    # After resume: runtime should have merged resume value into state (user_decision, user_updated_text)
    # Defensive: ensure update_attempts key exists
    state.setdefault("_user_update_attempts", 0)

    # If user decided update, apply update logic
    if state.get("user_decision") == "update":
        state["_user_update_attempts"] += 1
        MAX_ATTEMPTS = 5
        new_text = state.get("user_updated_text")

        if state["_user_update_attempts"] > MAX_ATTEMPTS:
            state.setdefault("warnings", []).append("user update attempts exceeded; auto-approving")
            state.setdefault("improvements", []).append({"action": "update_failed_max_attempts"})
            state["user_decision"] = "approve"
            state.pop("user_updated_text", None)
        else:
            if isinstance(new_text, str) and new_text.strip():
                state["text"] = new_text
                for k in ("chunks","entities","relations","bpmn","mapping","meta","render_snapshot","draft_snapshot"):
                    state.pop(k, None)
                state.setdefault("improvements", []).append({"action":"user_updated_text"})
                state.pop("user_updated_text", None)

    # After handling, optionally remove user_decision for cleanup OR leave it for router to read.
    # If your router reads user_decision, keep it until router runs; you can have a cleanup node later to pop it.

    return state
class InputData(BaseModel):
    text: str = "" # Raw input text describing the business process     
def build_graph_a():
    g = StateGraph(PipelineState)
    g.add_node("preprocess", node_preprocess)
    g.add_node("pos_dep", node_pos_dep)
    g.add_node("ner_llm", node_ner_llm)
    g.add_node("retrieve", node_retrieve)
    g.add_node("bpmn_map", node_bpmn_with_map)
    g.add_node("validate", node_validate)
    g.add_node("render", node_render)
    g.add_node("user_feedback", node_user_feedback)
    g.set_entry_point("preprocess")
    g.add_edge("preprocess", "pos_dep")
    g.add_edge("pos_dep", "ner_llm")
    g.add_edge("ner_llm", "retrieve")
    g.add_edge("retrieve", "bpmn_map")
    g.add_edge("bpmn_map", "validate")
    g.add_edge("validate", "render")
    g.add_edge("render", "user_feedback")

    g.add_conditional_edges(
        "user_feedback",
        apply_router,
        {
            "update_all": "preprocess",
            "approve": END
        },
    )
    
    return g.compile()

# Export a module-level compiled graph for langgraph dev
compiled_graph_a = build_graph_a()
