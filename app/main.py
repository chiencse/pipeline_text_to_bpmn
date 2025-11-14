from collections import defaultdict
from app.state import log_state
from app.tools.run_with_pretty_log import print_event, print_summary
from app.tools.viz_graphs import visualize_state
from fastapi import FastAPI
from pydantic import BaseModel
from app.scenario_a import build_graph_a
from app.scenario_b import build_graph_b
from app.utils import add_feedback, memory_get, memory_set
import json, time

app = FastAPI(title="BPMN Generator (Scenario A & B)")

graph_a = build_graph_a()
graph_b = build_graph_b()

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

@app.post("/pipeline/a")
def pipeline_a(req: Req):
    state = {"text": req.text, "config": req.options or {}}
    out = graph_a.invoke(state)
    # Log and visualize state after graph execution
    log_path = log_state(out, step="pipeline_a")
    visualize_state(out, step="pipeline_a", log_path=log_path)
    return out

@app.post("/pipeline/b")
def pipeline_b(req: Req):
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