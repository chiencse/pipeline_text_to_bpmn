from collections import defaultdict
from app.state import log_state
from app.logs.db_logger import create_pipeline_log, log_feedback_to_db, log_state_to_db
from app.tools.run_with_pretty_log import print_event, print_summary
from app.tools.viz_graphs import visualize_state
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.scenario_a import build_graph_a
from app.scenario_b import build_graph_b
from app.utils import add_feedback, memory_get, memory_set
import json, time
import uuid
import threading
from typing import Optional, Dict, Any, List

app = FastAPI(title="BPMN Generator (Scenario A & B)")

# Add CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

graph_a = build_graph_a()
graph_b = build_graph_b()

# Store pending interrupts: thread_id -> interrupt_data
pending_interrupts: Dict[str, Dict[str, Any]] = {}

# Helper functions for background logging with error handling
def log_feedback_to_db_safe(*args, **kwargs):
    """Wrapper for log_feedback_to_db with error handling"""
    try:
        log_feedback_to_db(*args, **kwargs)
    except Exception as e:
        print(f"Warning: Failed to log feedback to database in background: {e}")

def create_pipeline_log_safe(thread_id: str, text: str):
    """Wrapper for create_pipeline_log with error handling"""
    try:
        create_pipeline_log(thread_id, text)
    except Exception as e:
        print(f"Warning: Failed to create pipeline log in background: {e}")

class Req(BaseModel):
    text: str
    options: dict | None = None

class FeedbackReq(BaseModel):
    node_id: str
    selected_activity_id: str
    reason: str | None = None

class MemoryReq(BaseModel):
    key: str
    value: dict

class FeedbackResponse(BaseModel):
    user_decision: Optional[str] = None
    user_updated_text: Optional[str] = None
    user_feedback_text: Optional[str] = None
    user_mapping_decision: Optional[str] = None
    user_mapping_feedback_text: Optional[str] = None
    selected_node_ids: Optional[List[str]] = None  # New field for mapping feedback

@app.post("/pipeline/a")
def pipeline_a(req: Req):
    state = {"text": req.text, "config": req.options or {}}
    out = graph_a.invoke(state)
    # Log and visualize state after graph execution
    log_path = log_state(out, step="pipeline_a")
    visualize_state(out, step="pipeline_a", log_path=log_path)
    return out

@app.post("/pipeline/b/start")
def pipeline_b_start(req: Req):
    """
    Start Scenario B pipeline with a new thread_id.
    Returns thread_id and initial state. Use this thread_id for subsequent feedback calls.
    
    When interrupt() is called in a node:
    - Graph execution pauses immediately
    - Stream stops yielding events
    - State is saved to checkpoint with .next containing the waiting node
    """
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = {"text": req.text, "config": req.options or {}}
    threading.Thread(
                target=create_pipeline_log_safe,
                args=(thread_id, req.text),
                daemon=True
            ).start()
    try:
        # Stream through the graph execution
        # When interrupt() is called, stream will yield '__interrupt__' event and stop
        event_count = 0
        interrupt_event = None
        last_node_state = None
        
        for event in graph_b.stream(initial_state, config, stream_mode="updates"):
            event_count += 1
            event_keys = list(event.keys())
            print(f"Event {event_count}: {event_keys}")
            
            # Check for interrupt event
            if "__interrupt__" in event:
                interrupt_e = event["__interrupt__"]

                # interrupt_event là tuple -> lấy phần tử đầu
                interrupt_event = interrupt_e[0].value

                print("🔴 Interrupt payload (JSON):")
                print(interrupt_event)
            
            # Capture last node state before interrupt (should have markers)
            if event_keys and event_keys[0] != "__interrupt__":
                last_node_state = event.get(event_keys[0], {})
                # Check if this node has interrupt markers
                if isinstance(last_node_state, dict) and "__pending_interrupt__" in last_node_state:
                    print(f"✅ Found interrupt markers in node {event_keys[0]}")
        
        print(f"Stream completed with {event_count} events")
        
        # After stream completes, check checkpoint state
        checkpoint_state = graph_b.get_state(config)
        
        print(f"Checkpoint state: next={checkpoint_state.next if checkpoint_state else None}")
        
        if checkpoint_state and checkpoint_state.next:
            # There's a pending node - graph is waiting (interrupted)
            state_values = checkpoint_state.values or {}
            pending_node = checkpoint_state.next[0]
            
            # Try multiple sources for interrupt payload:
            # 1. From interrupt event (most reliable)
            interrupt_payload = None
            if interrupt_event:
                interrupt_payload = interrupt_event if isinstance(interrupt_event, dict) else {"message": str(interrupt_event)}
            
            # 2. From last node state before interrupt
            if not interrupt_payload and last_node_state:
                interrupt_payload = last_node_state.get("__pending_interrupt__")
            
            # 3. From checkpoint state values
            if not interrupt_payload:
                interrupt_payload = state_values.get("__pending_interrupt__")
            
            # 4. Build from state if node is a feedback node
            if not interrupt_payload:
                if pending_node == "user_feedback":
                    interrupt_payload = {
                        "type": "bpmn_feedback",
                        "instruction": "Review the generated BPMN structure. Do you approve it?",
                        "bpmn": state_values.get("bpmn", {}),
                        "draft_snapshot": state_values.get("draft_snapshot", {})
                    }
                elif pending_node == "user_feedback_mapping":
                    interrupt_payload = {
                        "type": "mapping_feedback",
                        "instruction": "Review the activity candidate mappings. Do you approve them?",
                        "bpmn": state_values.get("bpmn", {}),
                        "mapping": state_values.get("mapping", [])
                    }
            
            interrupt_node = state_values.get("__interrupt_node__") or pending_node
            
            if interrupt_payload:
                # Store interrupt info for future queries
                pending_interrupts[thread_id] = {
                    "payload": interrupt_payload,
                    "node": interrupt_node,
                    "state": state_values
                }
                return {
                    "thread_id": thread_id,
                    "status": "waiting_feedback",
                    "interrupt": interrupt_payload,
                    "state": state_values
                }
            else:
                # Node is pending but no interrupt payload found
                print(f"Warning: Node {pending_node} is pending but no interrupt payload found")
                print(f"State keys: {list(state_values.keys())}")
                return {
                    "thread_id": thread_id,
                    "status": "paused",
                    "next_node": pending_node,
                    "state": state_values,
                    "message": "Pipeline paused, but no interrupt payload found. May need to resume manually."
                }
        
        # No pending nodes - pipeline completed successfully
        final_state = graph_b.get_state(config)
        
        return {
            "thread_id": thread_id,
            "status": "completed",
            "state": final_state.values if final_state.values else {}
        }
    except Exception as e:
        print(f"Exception during pipeline execution: {type(e).__name__}: {str(e)}")
        # Check if graph is in interrupted state
        checkpoint_state = graph_b.get_state(config)
        if checkpoint_state and checkpoint_state.next:
            state_values = checkpoint_state.values or {}
            interrupt_payload = state_values.get("__pending_interrupt__")
            
            pending_interrupts[thread_id] = {
                "payload": interrupt_payload or {"message": "Pipeline paused for user feedback"},
                "node": checkpoint_state.next[0] if checkpoint_state.next else "unknown",
                "state": state_values
            }
            # Log to database in background (non-blocking)
            threading.Thread(
                target=create_pipeline_log_safe,
                args=(thread_id, req.text),
                daemon=True
            ).start()
            return {
                "thread_id": thread_id,
                "status": "waiting_feedback",
                "interrupt": interrupt_payload or {"message": "Pipeline paused for user feedback"},
                "state": state_values
            }
        raise HTTPException(status_code=500, detail=f"Pipeline execution error: {str(e)}")

@app.get("/pipeline/b/feedback/{thread_id}")
def get_pending_feedback(thread_id: str):
    """
    Get pending feedback request for a thread_id.
    Frontend should poll this endpoint to check if feedback is needed.
    """
    config = {"configurable": {"thread_id": thread_id}}
    checkpoint_state = graph_b.get_state(config)
    
    if checkpoint_state is None:
        raise HTTPException(status_code=404, detail="Thread not found")
    
    # Check if there's a pending interrupt
    if thread_id in pending_interrupts:
        interrupt_info = pending_interrupts[thread_id]
        return {
            "status": "waiting_feedback",
            "interrupt": interrupt_info["payload"],
            "node": interrupt_info["node"],
            "state": interrupt_info.get("state", checkpoint_state.values or {})
        }
    
    # Check checkpoint state
    if checkpoint_state.next:
        # There are pending nodes - check for interrupt in state
        state_values = checkpoint_state.values or {}
        interrupt_payload = state_values.get("__pending_interrupt__")
        interrupt_node = state_values.get("__interrupt_node__")
        
        if interrupt_payload:
            # Store interrupt info if not already stored
            if thread_id not in pending_interrupts:
                pending_interrupts[thread_id] = {
                    "payload": interrupt_payload,
                    "node": interrupt_node or (checkpoint_state.next[0] if checkpoint_state.next else "unknown"),
                    "state": state_values
                }
            return {
                "status": "waiting_feedback",
                "interrupt": interrupt_payload,
                "node": interrupt_node or "unknown",
                "state": state_values
            }
        else:
            return {
                "status": "running",
                "current_node": checkpoint_state.next[0] if checkpoint_state.next else None,
                "message": "Pipeline is still running"
            }
    else:
        return {
            "status": "completed",
            "state": checkpoint_state.values if checkpoint_state.values else {}
        }

@app.post("/pipeline/b/feedback/{thread_id}")
def submit_feedback(thread_id: str, feedback: FeedbackResponse, background_tasks: BackgroundTasks):
    """
    Submit user feedback to resume the pipeline.
    The feedback should match the interrupt payload structure.
    """
    config = {"configurable": {"thread_id": thread_id}}
    checkpoint_state = graph_b.get_state(config)
    
    if checkpoint_state is None:
        raise HTTPException(status_code=404, detail="Thread not found")
    
    # Determine which feedback node we're resuming
    # Check the pending node from checkpoint, not just state values
    state_values = checkpoint_state.values or {}
    pending_node = checkpoint_state.next[0] if checkpoint_state.next else None
    
    update = {}

    
    # Determine which feedback node is pending based on checkpoint.next
    if pending_node == "user_feedback":
        # First feedback (BPMN) is pending
        print(f"📝 Resuming BPMN feedback (user_feedback)")
        update["user_decision"] = feedback.user_decision or "approve"
        
        # Log feedback to database in background (non-blocking)
        if thread_id:
            background_tasks.add_task(
                log_feedback_to_db_safe,
                pipeline_id=thread_id,
                node_ids=feedback.selected_node_ids if feedback.selected_node_ids else [],
                user_decision=feedback.user_decision or "approve",
                is_mapping=False,
                user_feedback_text=feedback.user_feedback_text,
                bpmn=state_values.get("bpmn"),
                mapping=None
            )
        
        if feedback.user_feedback_text:
            update["user_feedback_text"] = feedback.user_feedback_text
        if feedback.user_updated_text:
            update["user_updated_text"] = feedback.user_updated_text
        if feedback.selected_node_ids:
            update["selected_node_ids"] = feedback.selected_node_ids
            
    elif pending_node == "user_feedback_mapping":
        # Second feedback (Mapping) is pending
        print(f"📝 Resuming Mapping feedback (user_feedback_mapping)")
        update["user_mapping_decision"] = feedback.user_mapping_decision or "approve"
        
        # Log feedback to database in background (non-blocking)
        if thread_id:
            selected_node_ids = feedback.selected_node_ids or []
            background_tasks.add_task(
                log_feedback_to_db_safe,
                pipeline_id=thread_id,
                node_ids=selected_node_ids,
                user_decision=feedback.user_mapping_decision or "approve",
                is_mapping=True,
                user_feedback_text=feedback.user_mapping_feedback_text,
                bpmn=state_values.get("bpmn"),
                mapping=state_values.get("mapping"),
                node_mapping_feedback=selected_node_ids
            )
        
        if feedback.user_mapping_feedback_text:
            update["user_mapping_feedback_text"] = feedback.user_mapping_feedback_text
        if feedback.selected_node_ids:
            update["selected_node_ids"] = feedback.selected_node_ids
    else:
        # Fallback: check state values if node name doesn't match
        print(f"⚠️ Warning: Unknown pending node '{pending_node}', falling back to state check")
        if "user_decision" not in state_values or not state_values.get("user_decision"):
            # First feedback (BPMN)
            update["user_decision"] = feedback.user_decision or "approve"
            if feedback.user_feedback_text:
                update["user_feedback_text"] = feedback.user_feedback_text
            if feedback.user_updated_text:
                update["user_updated_text"] = feedback.user_updated_text
        elif "user_mapping_decision" not in state_values or not state_values.get("user_mapping_decision"):
            # Second feedback (Mapping)
            update["user_mapping_decision"] = feedback.user_mapping_decision or "approve"
            if feedback.user_mapping_feedback_text:
                update["user_mapping_feedback_text"] = feedback.user_mapping_feedback_text
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Cannot determine which feedback to resume. Pending node: {pending_node}, State keys: {list(state_values.keys())}"
            )
    
    try:
        # IMPORTANT: To truly resume, we must update the checkpointed state,
        # then call stream() with NO new input. Passing `update` as the first
        # argument would restart the graph from START with a new input.
        #
        # This call merges `update` into the last checkpoint state.
        graph_b.update_state(config, update)

        # Now resume execution from the pending node using the updated state.
        event_count = 0
        resume_interrupt_event = None
        resume_last_node_state = None

        for event in graph_b.stream(None, config, stream_mode="updates"):
            event_count += 1
            event_keys = list(event.keys())
            print(f"Resume event {event_count}: {event_keys}")

            # Check for interrupt event in resume stream
            if "__interrupt__" in event:
                interrupt_e = event["__interrupt__"]
                # interrupt_event là tuple -> lấy phần tử đầu
                resume_interrupt_event = interrupt_e[0].value
                print("🔴 Resume interrupt payload (JSON):")
                print(resume_interrupt_event)

            # Capture last node state before interrupt
            if event_keys and event_keys[0] != "__interrupt__":
                resume_last_node_state = event.get(event_keys[0], {})
                if isinstance(resume_last_node_state, dict) and "__pending_interrupt__" in resume_last_node_state:
                    print(f"✅ Found interrupt markers in resume node {event_keys[0]}")

        print(f"Resume stream completed with {event_count} events")

        # Check if there's another interrupt
        checkpoint_state = graph_b.get_state(config)

        print(f"Resume checkpoint state: next={checkpoint_state.next if checkpoint_state else None}")

        if checkpoint_state and checkpoint_state.next:
            # Graph is waiting at another node (interrupted again)
            state_values = checkpoint_state.values or {}
            pending_node = checkpoint_state.next[0]

            # Try multiple sources for interrupt payload (same as start):
            # 1. From interrupt event (most reliable)
            interrupt_payload = None
            if resume_interrupt_event:
                interrupt_payload = (
                    resume_interrupt_event
                    if isinstance(resume_interrupt_event, dict)
                    else {"message": str(resume_interrupt_event)}
                )

            # 2. From last node state before interrupt
            if not interrupt_payload and resume_last_node_state:
                interrupt_payload = resume_last_node_state.get("__pending_interrupt__")

            # 3. From checkpoint state values
            if not interrupt_payload:
                interrupt_payload = state_values.get("__pending_interrupt__")

            # 4. Build from state if node is a feedback node
            if not interrupt_payload:
                if pending_node == "user_feedback":
                    interrupt_payload = {
                        "type": "bpmn_feedback",
                        "instruction": "Review the generated BPMN structure. Do you approve it?",
                        "bpmn": state_values.get("bpmn", {}),
                        "draft_snapshot": state_values.get("draft_snapshot", {}),
                    }
                elif pending_node == "user_feedback_mapping":
                    interrupt_payload = {
                        "type": "mapping_feedback",
                        "instruction": "Review the activity candidate mappings. Do you approve them?",
                        "bpmn": state_values.get("bpmn", {}),
                        "mapping": state_values.get("mapping", []),
                    }

            interrupt_node = state_values.get("__interrupt_node__") or pending_node

            if interrupt_payload:
                # Store new interrupt info
                pending_interrupts[thread_id] = {
                    "payload": interrupt_payload,
                    "node": interrupt_node,
                    "state": state_values,
                }
                print(f"✅ New interrupt detected after resume: {interrupt_payload.get('type', 'unknown')}")
                return {
                    "thread_id": thread_id,
                    "status": "waiting_feedback",
                    "interrupt": interrupt_payload,
                    "state": state_values,
                }
            else:
                print(
                    f"⚠️ Warning: Node {pending_node} is pending but no interrupt payload found after resume"
                )

        # Pipeline completed
        if thread_id in pending_interrupts:
            del pending_interrupts[thread_id]

        final_state = graph_b.get_state(config)
        state_values = final_state.values or {}
        return {
            "thread_id": thread_id,
            "status": "completed",
            "state": state_values,
            "mapping": state_values.get("mapping", []),
            "bpmn": state_values.get("bpmn", {}),
        }
    except Exception as e:
        print(f"Exception during resume: {type(e).__name__}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Pipeline resume error: {str(e)}")

@app.get("/pipeline/b/status/{thread_id}")
def get_pipeline_status(thread_id: str):
    """
    Get current status and state of a pipeline thread.
    """
    config = {"configurable": {"thread_id": thread_id}}
    checkpoint_state = graph_b.get_state(config)
    
    if checkpoint_state is None:
        raise HTTPException(status_code=404, detail="Thread not found")
    
    if thread_id in pending_interrupts:
        return {
            "status": "waiting_feedback",
            "interrupt": pending_interrupts[thread_id]["payload"],
            "node": pending_interrupts[thread_id]["node"]
        }
    
    if checkpoint_state.next:
        return {
            "status": "running",
            "current_node": checkpoint_state.next[0] if checkpoint_state.next else None,
            "state": checkpoint_state.values if checkpoint_state.values else {}
        }
    
    return {
        "status": "completed",
        "state": checkpoint_state.values if checkpoint_state.values else {}
    }

@app.post("/pipeline/b")
def pipeline_b(req: Req):
    """
    Legacy endpoint: Run Scenario B pipeline synchronously (without interrupts).
    For new implementations, use /pipeline/b/start with feedback endpoints.
    """
    state = {"text": req.text, "config": req.options or {}}
    out = graph_b.invoke(state)
    # Log and visualize state after graph execution
    log_path = log_state(out, step="pipeline_b")
    visualize_state(out, step="pipeline_b", log_path=log_path)
    return out

@app.post("/feedback/mapping")
def feedback_mapping(fb: FeedbackReq):
    add_feedback({"node_id": fb.node_id, "activity_id": fb.selected_activity_id, "reason": fb.reason})
    # TODO: tăng rule_bonus vào memory_kv nếu muốn tái xếp hạng tốt hơn
    return {"ok": True}

@app.get("/memory/{key}")
def get_mem(key: str):
    return {"key": key, "value": memory_get(key)}

@app.post("/memory")
def set_mem(req: MemoryReq):
    memory_set(req.key, req.value)
    return {"ok": True}

# if __name__ == "__main__":
#     compiled = build_graph_a()
#     state = {"text": "Send email with Gmail after creating sheet."}

#     def pretty_print_state(state_obj, indent=4):
#         """Hiển thị dict/lists lồng nhau dễ đọc"""
#         if isinstance(state_obj, (dict, list)):
#             print(json.dumps(state_obj, indent=indent, ensure_ascii=False))
#         else:
#             print(state_obj)

#     for event in compiled.stream(state):
#         if not isinstance(event, dict):
#             print(f"\n⚙️ EVENT (non-dict): {event}")
#             continue

#         # Mỗi event chỉ có 1 node key duy nhất
#         node_name, node_data = next(iter(event.items()))
#         print(f"\n🧩 NODE: {node_name}")
#         print("──────────────────────────────")
#         if isinstance(node_data, dict):
#             for key, value in node_data.items():
#                 print(f"  🔸 {key}:")
#                 if isinstance(value, (dict, list)):
#                     pretty_print_state(value, indent=8)
#                 else:
#                     print(f"        {value}")
#         else:
#             pretty_print_state(node_data)
            
if __name__ == "__main__":
    compiled = build_graph_b()

    # state vào thử nghiệm
    state = {"text": "Send email with Gmail after creating sheet."}
    from rich.console import Console
    console=Console()
    from rich.panel import Panel
    node_start_ts = {}
    node_durations = defaultdict(float)
    # đánh dấu start time cho mỗi node ngay khi thấy lần đầu
    for event in compiled.stream(state):
      # mỗi event là dict {node_name: state_snapshot}
      if not isinstance(event, dict) or not event:
          console.print(Panel(str(event), title="Event", border_style="white"))
          continue

      node_name, node_data = next(iter(event.items()))
      # nếu chưa có start ts -> set ngay
      node_start_ts.setdefault(node_name, time.time())
      print_event(node_name, node_data)

    print_summary()