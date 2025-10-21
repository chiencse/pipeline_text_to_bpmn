from langgraph.graph import StateGraph, END
from app.state import PipelineState
from app.utils import simple_chunk
from app.ner_hf import run_hf_ner
from app.relation_extraction import extract_relations_spacy
from app.retrieval import hybrid_search
from app.mapper import post_map_task, auto_fill_bindings
from app.validator import validate_bpmn
from app.llm import call_llm_bpmn_free

def node_preprocess(state: PipelineState):
    state["chunks"] = simple_chunk(state["text"])
    return state

def node_ner_ft(state: PipelineState):
    state["entities"] = run_hf_ner(state["chunks"])
    return state

def node_re_rule(state: PipelineState):
    state["relations"] = extract_relations_spacy(state["text"], state["entities"])
    return state

def node_bpmn_free(state: PipelineState):
    out = call_llm_bpmn_free(state["text"], state["entities"], state["relations"])
    state["bpmn"] = out.bpmn
    state["entities"] = [e.dict() for e in out.entities]
    state["relations"] = [r.dict() for r in out.relations]
    return state

def retrieve_candidates_for_task(task, entities):
    # dùng task name + SYS/DOC làm query
    ctx = " ".join([e["text"] for e in entities if e["label"] in ("SYS","DOC","ACT")])
    q = f'{task["name"]} {ctx}'.strip()
    return hybrid_search(q, k=5)

def node_retrieve_map(state: PipelineState):
    tasks = [n for n in state["bpmn"]["nodes"] if n["type"] in ("Task","UserTask","ManualTask")]
    mapping = []
    for t in tasks:
        topk = retrieve_candidates_for_task(t, state["entities"])
        m = post_map_task(t, topk)
        # điền bindings đơn giản theo requiredArgs của best (nếu có)
        if topk and m.get("type") == "ServiceTask":
            best = next((c for c in topk if c["activity_id"] == m["activity_id"]), None)
            if best:
                m = auto_fill_bindings(m, best.get("requiredArgs", []), state["entities"])
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
    g.add_node("ner_ft", node_ner_ft)
    g.add_node("re_rule", node_re_rule)
    g.add_node("bpmn_free", node_bpmn_free)
    g.add_node("retrieve_map", node_retrieve_map)
    g.add_node("validate", node_validate)
    g.add_node("render", node_render)
    g.set_entry_point("preprocess")
    g.add_edge("preprocess","ner_ft")
    g.add_edge("ner_ft","re_rule")
    g.add_edge("re_rule","bpmn_free")
    g.add_edge("bpmn_free","retrieve_map")
    g.add_edge("retrieve_map","validate")
    g.add_edge("validate","render")
    g.add_edge("render", END)
    return g.compile()

# Export a module-level compiled graph for langgraph dev
compiled_graph_b = build_graph_b()
