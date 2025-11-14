from langgraph.graph import StateGraph, END
from app.pos_dep import node_pos_dep
from app.preprocessing import node_preprocess
from app.state import PipelineState
from app.utils import simple_chunk
from app.ner_hf import run_hf_ner
from app.relation_extraction import extract_relations_from_state
from app.retrieval import hybrid_search
from app.mapper import post_map_task, auto_fill_bindings
from app.validator import validate_bpmn
from app.llm import call_llm_bpmn_free

# def node_preprocess(state: PipelineState):
#     state["chunks"] = simple_chunk(state["text"])
#     return state

def node_ner_ft(state: PipelineState):
    state["entities"] = run_hf_ner(state["chunks"])
    return state

def node_re_rule(state: PipelineState):
    state["relations"] = extract_relations_spacy(state["text"], state["entities"])
    return state

def node_bpmn_free(state: PipelineState):
    out = call_llm_bpmn_free(state["text"], state["entities"], state["relations"])
    state["bpmn"] = out.bpmn
    # state["entities"] = [e.dict() for e in out.entities]
    state["relations"] = [r.dict() for r in out.relations]
    return state

def retrieve_candidates_for_task(task, entities):
    # dùng task name + SYS/DOC làm query
    # ctx = " ".join([e["text"] for e in entities if e["label"] in ("SYS","DOC","ACT")])
    q = f'{task["name"]}'.strip()
    return hybrid_search(q, k=5)

def node_retrieve_map(state: PipelineState):
    # Lấy danh sách task hợp lệ (có type phù hợp và là dict)
    nodes = state.get("bpmn", {}).get("nodes", []) or []
    tasks = [n for n in nodes if isinstance(n, dict) and n.get("type") in ("Task", "UserTask", "ManualTask")]

    mapping = []
    entities = state.get("entities", {})

    for t in tasks:
        # Phòng thủ: nếu retrieve trả None thì coi như []
        topk = retrieve_candidates_for_task(t, entities) or []
        # Bảo đảm topk là list các dict
        if not isinstance(topk, list):
            topk = []

        m = post_map_task(t, topk)

        # Nếu m không phải dict, ghi nhận "unmapped" và bỏ qua auto-fill
        if not isinstance(m, dict):
            mapping.append({
                "type": "Unmapped",
                "reason": f"post_map_task returned {type(m).__name__}",
                "raw": m,
                "source_task_id": t.get("id"),
                "source_task_name": t.get("name"),
            })
            continue

        # Chỉ auto-fill khi là ServiceTask và có activity_id
        if topk and m.get("type") == "ServiceTask":
            act_id = m.get("activity_id")
            if act_id is not None:
                # Tìm "best" an toàn: chỉ xét phần tử là dict và có activity_id
                best = next(
                    (c for c in topk if isinstance(c, dict) and c.get("activity_id") == act_id),
                    None
                )
            else:
                best = None

            # Chỉ gọi .get khi best là dict
            if isinstance(best, dict):
                required_args = best.get("requiredArgs") or []
                # Đảm bảo required_args là list
                if not isinstance(required_args, list):
                    required_args = []
                try:
                    m = auto_fill_bindings(m, required_args, entities)
                except Exception as e:
                    # Không làm hỏng toàn bộ pipeline; ghi lý do vào mapping item
                    m.setdefault("_auto_fill_error", str(e))

        mapping.append(m)

    state["mapping"] = mapping
    return state


def node_validate(state: PipelineState):
    errors = validate_bpmn(state["bpmn"], scenario="B")
    state["meta"] = {"errors": errors}
    return state

def node_render(state: PipelineState):
    return state

def build_graph_b():
    g = StateGraph(PipelineState)
    g.add_node("preprocess", node_preprocess)
    g.add_node("pos_dep", node_pos_dep)
    g.add_node("ner_ft", node_ner_ft)
    g.add_node("re_rule", extract_relations_from_state)
    g.add_node("bpmn_free", node_bpmn_free)
    g.add_node("retrieve_map", node_retrieve_map)
    g.add_node("validate", node_validate)
    g.add_node("render", node_render)
    g.set_entry_point("preprocess")
    g.add_edge("preprocess","pos_dep")
    g.add_edge("pos_dep","ner_ft")
    g.add_edge("ner_ft","re_rule")
    g.add_edge("re_rule","bpmn_free")
    g.add_edge("bpmn_free","retrieve_map")
    g.add_edge("retrieve_map","validate")
    g.add_edge("validate","render")
    g.add_edge("render", END)
    return g.compile()

# Export a module-level compiled graph for langgraph dev
compiled_graph_b = build_graph_b()
