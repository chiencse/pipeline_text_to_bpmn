from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver
from typing import Dict, List
import os

from app.pos_dep import node_pos_dep
from app.preprocessing import node_preprocess
from app.state import PipelineState
from app.utils import simple_chunk

from app.relation_extraction import extract_relations_from_state
from app.retrieval import hybrid_search, get_control_and_data_manipulation_templates
from app.validator import validate_bpmn
from app.llm import call_llm_bpmn_free, call_llm_evaluate_automation_feasibility, call_llm_bpmn_with_feedback, call_llm_mapping_with_feedback
from app.bpmn_converter import render_bpmn_output
from app.logs.db_logger import log_retrieval_scores_async
import json
from pprint import pprint

# ---------- Sanitizer helpers (prevent circular / non-serializable objects) ----------
def safe_serialize_obj(obj):
    """
    Conservative serializer: returns only JSON-safe primitives,
    recurses on lists/tuples/dicts, uses .dict()/to_dict() if available,
    otherwise returns a short repr.
    """
    try:
        if obj is None:
            return None
        if isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, (list, tuple, set)):
            return [safe_serialize_obj(x) for x in obj]
        if isinstance(obj, dict):
            return {str(k): safe_serialize_obj(v) for k, v in obj.items()}
        # dataclass/pydantic
        dict_method = getattr(obj, "dict", None) or getattr(obj, "to_dict", None)
        if callable(dict_method):
            try:
                return safe_serialize_obj(dict_method())
            except Exception:
                pass
        # Fallback: short repr
        r = repr(obj)
        return r if len(r) <= 512 else r[:512] + "..."
    except RecursionError:
        return "<recursive>"
    except Exception:
        return f"<unserializable {type(obj).__name__}>"

def sanitize_state_for_persistence(state: dict, allowed_keys=None):
    """
    Return a new dict with only allowed_keys serialized via safe_serialize_obj.
    If allowed_keys is None, will serialize all keys (may be heavy).
    """
    allowed_keys = set(allowed_keys) if allowed_keys is not None else None
    out = {}
    for k, v in state.items():
        if allowed_keys is not None and k not in allowed_keys:
            continue
        out[k] = safe_serialize_obj(v)
    return out

# ---------- Nodes ----------


def node_re_rule(state: PipelineState):
    state = extract_relations_from_state(state)
    return state

def node_bpmn_free(state: PipelineState):
    """
    Call LLM but store only serializable pieces.
    Avoid storing the entire 'out' object (which may contain callbacks/tracers).
    
    Checks if this is a feedback flow (user rejected previous BPMN):
    - If user_feedback_text exists, calls call_llm_bpmn_with_feedback
    - Otherwise, calls call_llm_bpmn_free normally
    """
    # Check if this is a feedback flow
    user_feedback_text = state.get("user_feedback_text", "")
    selected_node_ids = state.get("selected_node_ids", [])
    is_feedback_flow = bool(user_feedback_text and user_feedback_text.strip())
    
    if is_feedback_flow:
        # This is a feedback flow - call feedback function
        current_bpmn = state.get("bpmn", {})
        original_text = state.get("text", "")
        
        print(f"[node_bpmn_free] Feedback flow detected. Using call_llm_bpmn_with_feedback")
        print(f"[node_bpmn_free] Feedback text: {user_feedback_text[:100]}...")
        print(f"[node_bpmn_free] Selected node IDs: {selected_node_ids}")
        
        out = call_llm_bpmn_with_feedback(
            original_text=original_text,
            current_bpmn=current_bpmn,
            user_feedback_text=user_feedback_text,
            selected_node_ids=selected_node_ids if selected_node_ids else None
        )
        
        # Clear feedback text after using it
        state["user_decision"] = None
        state["user_feedback_text"] = None
        state["selected_node_ids"] = None
    else:
        # Normal flow - call regular function
        out = call_llm_bpmn_free(state.get("text", ""))
    # bpmn_safe: prefer .dict() if available
    try:
        bpmn_obj = getattr(out, "bpmn", None)
        if hasattr(bpmn_obj, "dict"):
            bpmn_safe = bpmn_obj.dict()
        else:
            bpmn_safe = safe_serialize_obj(bpmn_obj)
    except Exception:
        bpmn_safe = safe_serialize_obj(getattr(out, "bpmn", None))

    # relations_safe: list of dicts
    try:
        relations_raw = getattr(out, "relations", []) or []
        relations_safe = [r.dict() if hasattr(r, "dict") else safe_serialize_obj(r) for r in relations_raw]
    except Exception:
        relations_safe = safe_serialize_obj(getattr(out, "relations", None))

    state["bpmn"] = bpmn_safe
    state["relations"] = relations_safe

    # # Optional: small llm metadata only
    # summary = getattr(out, "summary", None) or getattr(out, "meta", None) or None
    # if summary is not None:
    #     state["llm_summary"] = safe_serialize_obj(summary)

    # # Keep a small sanitized snapshot for history/logging if you want
    # state["__sanitized_snapshot_for_history"] = sanitize_state_for_persistence(
    #     state,
    #     allowed_keys=["text", "bpmn", "relations", "entities", "mapping", "meta", "improvements"]
    # )
    # return state
    # state["bpmn"] = out.bpmn
    # state["relations"] = out.relations
    return state

def retrieve_candidates_for_task(task, state: PipelineState = None):
    """
    Retrieve candidates using hybrid search for a task.
    Enhances query with relevant sentences from state based on node name.
    Logs retrieval scores asynchronously to database.
    """
    node_name = task.get("name", "")
    query_activity = task.get("queryActivity", "")
    node_id = task.get("id", "")
    q = query_activity + " " + node_name
    
    # If state is provided, find relevant sentences and integrate into query context
    # if state:
    #     chunks = state.get("chunks", [])
    #     relevant_sentences = []
        
    #     # Find sentences that contain words from node name
    #     node_name_lower = node_name.lower()
    #     node_words = set(node_name_lower.split())
        
    #     # Collect sentences from all chunks that are relevant to the node name
    #     for chunk in chunks:
    #         sentences = chunk.get("sentences", [])
    #         for sent in sentences:
    #             sent_text = sent.get("text", "").lower()
    #             # Check if sentence contains any word from node name
    #             if any(word in sent_text for word in node_words if len(word) > 2):  # Only check words longer than 2 chars
    #                 relevant_sentences.append(sent.get("text", ""))
        
    #     # Limit to top 3 most relevant sentences to avoid too long query
    #     if relevant_sentences:
    #         # Use simple scoring: prefer sentences with more matching words
    #         scored_sentences = []
    #         for sent in relevant_sentences[:10]:  # Limit candidate sentences
    #             sent_lower = sent.lower()
    #             match_count = sum(1 for word in node_words if len(word) > 2 and word in sent_lower)
    #             scored_sentences.append((match_count, sent))
            
    #         # Sort by match count and take top 1
    #         scored_sentences.sort(key=lambda x: x[0], reverse=True)
    #         top_sentences = [sent for _, sent in scored_sentences[:1]]
            
    #         # Integrate sentences into query context
    #         if top_sentences:
    #             context = " ".join(top_sentences)
    #             q = f"{node_name} {context}".strip()
    
    # Retrieve candidates from hybrid search
    candidates = hybrid_search(q, k=5)
    
    # Log retrieval scores asynchronously if state is provided
    if candidates and state:
        # Get thread_id from state (added in pipeline_b_start)
        thread_id = state.get("thread_id", "unknown")
        
        try:
            log_retrieval_scores_async(
                thread_id=thread_id,
                node_id=node_id,
                node_name=node_name,
                candidates=candidates
            )
        except Exception as e:
            print(f"[retrieve_candidates_for_task] Warning: Failed to start retrieval scores logging for node {node_id}: {e}")
    
    return candidates

def node_retrieve_map(state: PipelineState):
    """
    Retrieve candidates and map tasks to activities using LLM evaluation.
    Uses LLM to evaluate automation feasibility for each node based on candidates.
    Incorporates user feedback text if available from previous rejection.
    
    Checks if this is a feedback flow (user rejected previous mapping):
    - If user_mapping_feedback_text exists, calls call_llm_mapping_with_feedback
    - Otherwise, uses normal evaluation flow
    """

    bpmn = state.get("bpmn") or {}
    nodes = bpmn.get("nodes", []) if isinstance(bpmn, dict) else []
    flows = bpmn.get("flows", []) if isinstance(bpmn, dict) else []
    original_text = state.get("text", "")

    # Check if this is a feedback flow
    user_mapping_feedback_text = state.get("user_mapping_feedback_text", "")
    selected_node_ids = state.get("selected_node_ids", [])
    is_feedback_flow = bool(user_mapping_feedback_text and user_mapping_feedback_text.strip())

    # Get control and data manipulation templates (always included)
    control_templates, data_manipulation_templates = get_control_and_data_manipulation_templates()
    
    # Build act_candidates dict: node_id -> list of candidates
    act_candidates_dict = {}
    
    for n in nodes:
        node_id = n.get("id")
        node_type = n.get("type", "")
        
        # Retrieve candidates using hybrid search
        retrieved_candidates = retrieve_candidates_for_task(n, state)
        candidates = [c for c in retrieved_candidates if c.get("score") > 0.5] or []
        # For gateway nodes or nodes in loop, add control templates
        if n.get("in_loop", True) or node_type in ("ExclusiveGateway", "ParallelGateway", "InclusiveGateway", "Gateway"):
            # Merge control templates with retrieved candidates
            all_candidates = candidates + control_templates
            # Remove duplicates based on activity_id
            seen_ids = set()
            unique_candidates = []
            for c in all_candidates:
                act_id = c.get("activity_id")
                if act_id and act_id not in seen_ids:
                    seen_ids.add(act_id)
                    unique_candidates.append(c)
            act_candidates_dict[node_id] = unique_candidates
        else:
            act_candidates_dict[node_id] = candidates
    # Add control templates for flows with conditions
    for flow in flows:
        if flow.get("condition"):
            source_id = flow.get("source")
            if source_id not in act_candidates_dict:
                act_candidates_dict[source_id] = []
            # Merge control templates
            existing = act_candidates_dict[source_id]
            existing_ids = {c.get("activity_id") for c in existing if c.get("activity_id")}
            for ctrl in control_templates:
                if ctrl.get("activity_id") not in existing_ids:
                    existing.append(ctrl)
    
    if is_feedback_flow:
        # This is a feedback flow - call feedback function
        print(f"[node_retrieve_map] Feedback flow detected. Using call_llm_mapping_with_feedback")
        print(f"[node_retrieve_map] Feedback text: {user_mapping_feedback_text[:100]}...")
        print(f"[node_retrieve_map] Selected node IDs: {selected_node_ids}")
        
        current_mapping_raw = state.get("mapping") or []
        print(f"[node_retrieve_map] Current mapping (raw): {current_mapping_raw}")
        
        # Convert current_mapping from [{node_id: mapping_entry}, ...] to [mapping_entry, ...]
        current_mapping = []
        if current_mapping_raw:  # Only process if mapping exists and is not None/empty
            for item in current_mapping_raw:
                if isinstance(item, dict):
                    # Extract the mapping_entry from the dict
                    for node_id, mapping_entry in item.items():
                        current_mapping.append(mapping_entry)
        
        print(f"[node_retrieve_map] Current mapping (converted): {len(current_mapping)} entries")
        # Flatten candidates dict into a list with node_id attached
        candidates_list = []
        for node_id, candidates in act_candidates_dict.items():
            for candidate in candidates:
                candidate_with_node = candidate.copy()
                candidate_with_node["node_id"] = node_id
                candidates_list.append(candidate_with_node)
        

        # Call feedback function
        updated_mapping = call_llm_mapping_with_feedback(
            original_text=original_text,
            current_bpmn=bpmn,
            current_mapping=current_mapping,
            candidates=candidates_list,
            user_feedback_text=user_mapping_feedback_text,
            selected_node_ids=selected_node_ids if selected_node_ids else None
        )
        
        # Convert back to expected format: [{node_id: mapping_entry}, ...]
        act_candidates = []
        for mapping_entry in updated_mapping:
            node_id = mapping_entry.get("node_id")
            if node_id:
                act_candidates.append({node_id: mapping_entry})
        
        # Clear feedback text after using it
        state["user_mapping_feedback_text"] = None
        state["selected_node_ids"] = None
        state["user_mapping_decision"] = None
        
    else:
        # Normal flow - use evaluation
        # Incorporate feedback text if available from user rejection (legacy support)
        feedback_text = state.get("retrieval_feedback_text", "")
        if feedback_text:
            original_text = f"{original_text}\n\nUser feedback: {feedback_text}"
            # Clear feedback text after using it
            state["retrieval_feedback_text"] = None
        
        # Call LLM to evaluate automation feasibility
        print(" Calling LLM to evaluate automation feasibility...")

        evaluations = call_llm_evaluate_automation_feasibility(
            nodes=nodes,
            flows=flows,
            act_candidates=act_candidates_dict,
            original_text=original_text
        )
        
        print("Evaluations:", evaluations)
        # Build mapping from evaluations
        # Create a map: node_id -> evaluation result
        evaluation_map = {eval_result.get("node_id"): eval_result for eval_result in evaluations}
        
        # Format mapping output
        act_candidates = []
        for n in nodes:
            node_id = n.get("id")
            evaluation = evaluation_map.get(node_id)
            
            if evaluation:
                # Node was evaluated by LLM
                candidates = act_candidates_dict.get(node_id, [])
                selected_activity_id = evaluation.get("selected_activity_id")
                
                # Find selected candidate details
                selected_candidate = None
                if selected_activity_id:
                    for c in candidates:
                        if c.get("activity_id") == selected_activity_id:
                            selected_candidate = c
                            break
                
                mapping_entry = {
                    "node_id": node_id,
                    "activity_id": selected_activity_id,
                    "confidence": evaluation.get("confidence", 0.0),
                    "is_automatic": evaluation.get("is_automatic", False),
                    "reasoning": evaluation.get("reasoning", ""),
                    "candidates": [
                        {
                            "activity_id": c.get("activity_id"),
                            "score": c.get("score", 0.0),
                            "pkg": c.get("pkg", ""),
                            "keyword": c.get("keyword", "")
                        }
                        for c in candidates[:5]  # Top 5 candidates
                    ],
                    "input_bindings": {},
                    "outputs": []
                }
                act_candidates.append({node_id: mapping_entry})
            else:
                # Node was not evaluated (LLM skipped it) - mark as not automatic
                candidates = act_candidates_dict.get(node_id, [])
                mapping_entry = {
                    "node_id": node_id,
                    "activity_id": None,
                    "confidence": 0.0,
                    "is_automatic": False,
                    "reasoning": "no suitable automation candidates found",
                    "candidates": [
                        {
                            "activity_id": c.get("activity_id"),
                            "score": c.get("score", 0.0),
                            "pkg": c.get("pkg", ""),
                            "keyword": c.get("keyword", "")
                        }
                        for c in candidates[:5]
                    ],
                    "input_bindings": {},
                    "outputs": []
                }
                act_candidates.append({node_id: mapping_entry})
    
    state["mapping"] = act_candidates
    return state

def node_user_feedback_mapping(state: PipelineState):
    """
    Second user feedback node: After retrieve_map, send activity candidate mappings to frontend.
    If accept: continue to validate
    If reject: go back to retrieve_map with user feedback text
    """
    print("=== USER FEEDBACK NODE (MAPPING) ===")
    
    # If user_mapping_decision not set, interrupt and wait for frontend response
    if "user_mapping_decision" not in state or not state.get("user_mapping_decision"):
        mapping_data = state.get("mapping", [])
        bpmn_data = state.get("bpmn", {})
        
        payload = {
            "type": "mapping_feedback",
            "instruction": "Review the activity candidate mappings for each BPMN node. Do you approve them?",
            "bpmn": safe_serialize_obj(bpmn_data),
            "mapping": safe_serialize_obj(mapping_data),
            "hint": "Reply with {'user_mapping_decision': 'approve'} or {'user_mapping_decision': 'reject', 'user_mapping_feedback_text': '...'}"
        }
        # Store interrupt payload in state so FastAPI can retrieve it
        # IMPORTANT: Set markers BEFORE calling interrupt() so they're in checkpoint
        state["__pending_interrupt__"] = payload
        state["__interrupt_node__"] = "user_feedback_mapping"
        
        print(f"[node_user_feedback_mapping] Calling interrupt() with payload type: {payload['type']}")
        print(f"[node_user_feedback_mapping] State has __pending_interrupt__: {'__pending_interrupt__' in state}")
        
        # Call interrupt - this pauses execution and checkpoints current state
        res = interrupt(payload)
        
        print(f"[node_user_feedback_mapping] Resumed from interrupt with response: {res}")
        
        # When execution resumes (after user submits feedback), we get here
        # Clear interrupt markers and process the response
        state.pop("__pending_interrupt__", None)
        state.pop("__interrupt_node__", None)
        
        state["user_mapping_decision"] = res.get("user_mapping_decision", "approve")
        state["user_mapping_feedback_text"] = res.get("user_mapping_feedback_text", "")
    else:
        print(f"[node_user_feedback_mapping] Skipping interrupt, user_mapping_decision already set: {state.get('user_mapping_decision')}")

    # Defensive: ensure update_attempts key exists
    state.setdefault("_user_mapping_update_attempts", 0)

    # If user decided reject, apply feedback logic
    if state.get("user_mapping_decision") == "reject":
        state["_user_mapping_update_attempts"] += 1
        MAX_ATTEMPTS = 5
        feedback_text = state.get("user_mapping_feedback_text", "")

        if state["_user_mapping_update_attempts"] > MAX_ATTEMPTS:
            state.setdefault("warnings", []).append("user mapping feedback attempts exceeded; auto-approving")
            state.setdefault("improvements", []).append({"action": "mapping_feedback_failed_max_attempts"})
            state["user_mapping_decision"] = "approve"
            state["user_mapping_feedback_text"] = None
        else:
            if isinstance(feedback_text, str) and feedback_text.strip():
                # Keep mapping for node_retrieve_map to use in feedback flow
                # Don't clear mapping here - node_retrieve_map needs it for call_llm_mapping_with_feedback
                # Mapping will be updated by node_retrieve_map after feedback processing
                state.setdefault("improvements", []).append({"action": "mapping_feedback_applied"})
                # Don't clear user_mapping_feedback_text here - let node_retrieve_map clear it after using it

    return state

def node_validate(state: PipelineState):
    errors = validate_bpmn(state.get("bpmn", {}), scenario="B")
    state["meta"] = {"errors": errors}
    return state
def write_file(content: str, filename: str):
    """Write content to a file with specified filename."""
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception as e:
        print(f"Error writing file {filename}: {str(e)}")
        return False
    return True
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

    write_file(result.get("xml"), "output_bpmn/bpmn.xml")

    
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

# --- draft render & user-feedback nodes ---
def node_render_draft(state: PipelineState):
    state["draft_snapshot"] = {
        "bpmn": state.get("bpmn"),
        "summary": state.get("render_snapshot", {}),
    }
    return state

# Nếu runtime Python của bạn cung cấp interrupt() that raises INTERRUPT,
# import tương ứng. Nếu interrupt() trả về giá trị khi resume, code này vẫn hợp lý.


def node_user_feedback(state: PipelineState):
    """
    First user feedback node: Send BPMN state to frontend and wait for response.
    If accept: continue to retrieve_map
    If reject: go back to bpmn_free with user feedback text
    """
    print("=== USER FEEDBACK NODE (BPMN) ===")
    
    # If user_decision not set, interrupt and wait for frontend response
    if "user_decision" not in state or not state.get("user_decision"):
        bpmn_data = state.get("bpmn", {})
        payload = {
            "type": "bpmn_feedback",
            "instruction": "Review the generated BPMN structure. Do you approve it?",
            "bpmn": safe_serialize_obj(bpmn_data),
            "draft_snapshot": state.get("draft_snapshot", {}),
            "hint": "Reply with {'user_decision': 'approve'} or {'user_decision': 'reject', 'user_feedback_text': '...'}"
        }
        # Store interrupt payload in state so FastAPI can retrieve it
        # IMPORTANT: Set markers BEFORE calling interrupt() so they're in checkpoint
        state["__pending_interrupt__"] = payload
        state["__interrupt_node__"] = "user_feedback"
        
        print(f"[node_user_feedback] Calling interrupt() with payload type: {payload['type']}")
        print(f"[node_user_feedback] State has __pending_interrupt__: {'__pending_interrupt__' in state}")
        
        # Call interrupt - this pauses execution and checkpoints current state
        # The state above (with markers) SHOULD be saved to checkpoint
        res = interrupt(payload)
        
        print(f"[node_user_feedback] Resumed from interrupt with response: {res}")
        
        # When execution resumes (after user submits feedback), we get here
        # Clear interrupt markers and process the response
        # Note: Markers are cleared AFTER resume, so they remain in checkpoint during wait
        state.pop("__pending_interrupt__", None)
        state.pop("__interrupt_node__", None)
        
        state["user_decision"] = res.get("user_decision", "approve")
        state["user_feedback_text"] = res.get("user_feedback_text", "")
    else:
        print(f"[node_user_feedback] Skipping interrupt, user_decision already set: {state.get('user_decision')}")

    # Defensive: ensure update_attempts key exists
    state.setdefault("_user_update_attempts", 0)

    # If user decided reject, apply feedback logic
    if state.get("user_decision") == "reject":
        state["_user_update_attempts"] += 1
        MAX_ATTEMPTS = 5
        feedback_text = state.get("user_feedback_text", "")

        if state["_user_update_attempts"] > MAX_ATTEMPTS:
            state.setdefault("warnings", []).append("user feedback attempts exceeded; auto-approving")
            state.setdefault("improvements", []).append({"action": "feedback_failed_max_attempts"})
            state["user_decision"] = "approve"
            state["user_feedback_text"] = None
        else:
            if isinstance(feedback_text, str) and feedback_text.strip():
                # Clear BPMN and downstream data to restart from bpmn_free
                # Keep entities and relations as they're still valid
                # Keep user_feedback_text for node_bpmn_free to use in feedback flow
                for k in ("bpmn", "mapping", "meta", "render_snapshot", "draft_snapshot"):
                    state[k] = None
                state.setdefault("improvements", []).append({"action": "user_feedback_applied"})
                # Don't clear user_feedback_text here - let node_bpmn_free clear it after using it

    return state


# --- final feedback after render ---
def node_user_feedback_final(state: PipelineState):
    dec = state.get("user_final_decision")
    if dec not in ("update", "approve"):
        state["user_final_decision"] = "approve"
    return state
    # Entry router: resume only if draft/bpmn exists, otherwise preprocess
def entry_router(state: PipelineState):
        try:
            if state.get("resume_mode"):
                if state.get("draft_snapshot") or state.get("bpmn"):
                    return "user_feedback"
            return "preprocess"
        except Exception:
            return "preprocess"
def final_router(state: PipelineState):
        v = state.get("user_final_decision")
        return "update" if v == "update" else "approve"
# ---------- Build graph with robust routers ----------
def build_graph_b():
    g = StateGraph(PipelineState)

    # Core nodes

    g.add_node("preprocess", node_preprocess)
    g.add_node("pos_dep", node_pos_dep)

    g.add_node("re_rule", node_re_rule)
    g.add_node("bpmn_free", node_bpmn_free)
    g.add_node("render_draft", node_render_draft)
    g.add_node("user_feedback", node_user_feedback)
    g.add_node("retrieve_map", node_retrieve_map)
    g.add_node("validate", node_validate)
    g.add_node("render", node_render)
    # g.add_node("user_feedback_final", node_user_feedback_final)



    # g.add_conditional_edges(
    #     START,
    #     entry_router,
    #     {
    #         "preprocess": "preprocess",
    #         "user_feedback": "user_feedback",
    #     },
    # )

    # Normal flow
    g.add_edge(START, "preprocess")
    g.add_edge("preprocess", "pos_dep")
    g.add_edge("pos_dep", "re_rule")
    g.add_edge("re_rule", "bpmn_free")

    g.add_edge("bpmn_free", "render_draft")

    g.add_edge("render_draft", "user_feedback")
    

    # Router for first user feedback -> return 'reject' (go back to bpmn_free) or 'approve' (continue)
    def apply_router(state: PipelineState):
        v = state.get("user_decision", "approve")
        print("=== APPLY ROUTER (BPMN) ===" , v)
        return "reject" if v == "reject" else "approve"

    g.add_conditional_edges(
        "user_feedback",
        apply_router,
        {
            "reject": "bpmn_free",  # Go back to bpmn_free with feedback (keeps entities/relations)
            "approve": "retrieve_map",
        },
    )

    g.add_edge("retrieve_map", "user_feedback_mapping")
    
    # Router for second user feedback (mapping) -> return 'reject' (go back to retrieve_map) or 'approve' (continue)
    def apply_mapping_router(state: PipelineState):
        v = state.get("user_mapping_decision", "approve")
        print("=== APPLY ROUTER (MAPPING) ===" , v)
        return "reject" if v == "reject" else "approve"
    
    g.add_node("user_feedback_mapping", node_user_feedback_mapping)
    
    g.add_conditional_edges(
        "user_feedback_mapping",
        apply_mapping_router,
        {
            "reject": "retrieve_map",  # Go back to retrieve_map with feedback
            "approve": "validate",
        },
    )
    g.add_edge("validate", "render")
    # g.add_edge("render", "user_feedback_final")

    

    # g.add_conditional_edges(
    #     "user_feedback_final",
    #     final_router,
    #     {
    #         "update": "preprocess",
    #         "approve": END,
    #     },
    # )

    # Add checkpoint memory for interrupt support
    memory = MemorySaver()
    # Only add checkpointer when NOT running in LangGraph API mode
    # LangGraph API handles persistence automatically, so custom checkpointers cause errors
    # 
    # Detection strategy:
    # - If LANGGRAPH_API_URL is set, we're in LangGraph API mode → no checkpointer
    # - Otherwise, assume local FastAPI mode → use checkpointer for interrupt support
    # - Can override with LANGGRAPH_USE_CHECKPOINTER=false to disable even locally
    
    is_langgraph_api = os.getenv("LANGGRAPH_API_URL") is not None
    force_no_checkpointer = os.getenv("LANGGRAPH_USE_CHECKPOINTER", "").lower() in ("0", "false", "no")
    
    if is_langgraph_api or force_no_checkpointer:
        # Running in LangGraph API mode or explicitly disabled - don't add custom checkpointer
        # The API will handle persistence automatically
        return g.compile()
    else:
        # Running locally with FastAPI - add MemorySaver for interrupt support
        memory = MemorySaver()
        return g.compile(checkpointer=memory)

# __main__.py
compiled_graph_b = build_graph_b()

def pretty(obj):
    """Safe pretty print JSON-like structure."""
    try:
        print(json.dumps(obj, indent=2, ensure_ascii=False))
    except Exception:
        pprint(obj)


def run_debug(text: str):
    """
    Run LangGraph in streaming mode to debug node-by-node execution.
    """
    print("\n================= START DEBUG PIPELINE =================\n")
    initial: PipelineState = {"text": text}

    print("Initial State:")
    pretty(initial)
    print("\n--------------------------------------------------------\n")

    # Run with streaming to inspect every node execution
    # compiled_graph_b = build_graph_b()
    step_id = 0
    try:
        for event in compiled_graph_b.stream(initial, stream_mode=["updates", "custom"]):
            step_id += 1

            for e in event:
                print(f"\n🔵 STEP {step_id}: Node executed → {e}")
      
            # node = event.get("name")
            # vals = event.get("values") or {}
            # print(f"\n🔵 STEP {step_id}: Node executed → {node}")
            # print("Changes:")
            # pretty(vals)

            print("\n--------------------------------------------------------\n")

    except Exception as e:
        print("\n❌ ERROR OCCURRED DURING EXECUTION")
        print(type(e).__name__, ":", str(e))
        print("\n========================================================\n")
        raise

    print("\n================= END DEBUG PIPELINE =================\n")


if __name__ == "__main__":
    print("🔥 LangGraph Debug Runner")
    print("==========================")

    test_text = """
    Send an email to finance, attach the quotation, wait for reply,
    then update the invoice in the SAP system.
    """

    run_debug(test_text)

