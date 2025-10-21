from langgraph.graph import StateGraph, END
from app.state import PipelineState
from app.utils import simple_chunk
from app.retrieval import build_query_from_entities, hybrid_search
from app.validator import validate_bpmn
from app.llm import call_llm_ner, call_llm_bpmn_with_mapping

def node_preprocess(state: PipelineState):
    state["chunks"] = simple_chunk(state["text"])
    return state

def node_ner_llm(state: PipelineState):
    state["entities"] = call_llm_ner(state["chunks"])
    return state

def node_retrieve(state: PipelineState):
    q = build_query_from_entities(state["entities"])
    state["candidates"] = hybrid_search(q, k=5) if q else []
    return state

def node_bpmn_with_map(state: PipelineState):
    out = call_llm_bpmn_with_mapping(
        text=state["text"], entities=state["entities"], candidates=state["candidates"]
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
    return state

def build_graph_a():
    g = StateGraph(PipelineState)
    g.add_node("preprocess", node_preprocess)
    g.add_node("ner_llm", node_ner_llm)
    g.add_node("retrieve", node_retrieve)
    g.add_node("bpmn_map", node_bpmn_with_map)
    g.add_node("validate", node_validate)
    g.add_node("render", node_render)
    g.set_entry_point("preprocess")
    g.add_edge("preprocess", "ner_llm")
    g.add_edge("ner_llm", "retrieve")
    g.add_edge("retrieve", "bpmn_map")
    g.add_edge("bpmn_map", "validate")
    g.add_edge("validate", "render")
    g.add_edge("render", END)
    return g.compile()

# Export a module-level compiled graph for langgraph dev
compiled_graph_a = build_graph_a()
