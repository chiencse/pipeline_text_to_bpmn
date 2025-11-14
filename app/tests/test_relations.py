# tests/test_relations.py
import pytest

from app.pos_dep import node_pos_dep            # nơi bạn đặt node_pos_dep
from app.relation_extraction import extract_relations_from_state  # file bạn chứa hàm mới
# Nếu khác path, chỉnh lại import phù hợp

def _span(text: str, sub: str):
    i = text.index(sub)
    return i, i + len(sub)

def test_agent_and_object_simple():
    # An officer selects a menu.
    sent = "An officer selects a menu."
    state = {"chunks": [{"text": sent, "start": 0, "end": len(sent)}]}
    state = node_pos_dep(state)

    # entities: officer (Role), menu (Object)
    os, oe = _span(sent, "officer")
    ms, me = _span(sent, "menu")
    state["entities"] = [
        {"text": "officer", "label": "Role", "start": os, "end": oe},
        {"text": "menu", "label": "Object", "start": ms, "end": me},
    ]

    rels = extract_relations_from_state(state)
    assert {"head": "selects", "tail": "officer", "type": "agent_of"} in rels
    assert {"head": "selects", "tail": "menu", "type": "object_of"} in rels

def test_exec_in_system_prep_in():
    # Create record in SAP.
    sent = "Create record in SAP."
    state = {"chunks": [{"text": sent, "start": 0, "end": len(sent)}]}
    state = node_pos_dep(state)

    ss, se = _span(sent, "SAP")
    state["entities"] = [{"text": "SAP", "label": "System", "start": ss, "end": se}]

    rels = extract_relations_from_state(state)
    # exec_in từ prep "in" + pobj "SAP"
    assert {"head": "Create", "tail": "SAP", "type": "exec_in"} in rels

def test_sequence_next_and_object_of():
    # Create and send email to the client.
    sent = "Create and send email to the client."
    state = {"chunks": [{"text": sent, "start": 0, "end": len(sent)}]}
    state = node_pos_dep(state)

    es, ee = _span(sent, "email")
    state["entities"] = [{"text": "email", "label": "Resource", "start": es, "end": ee}]

    rels = extract_relations_from_state(state)
    # sequence_next giữa 2 động từ "Create" -> "send"
    assert {"head": "Create", "tail": "send", "type": "sequence_next"} in rels
    # object_of cho send -> email
    assert {"head": "send", "tail": "email", "type": "object_of"} in rels
    # (hàm không sinh receiver_of cho "to the client", nên không assert phần đó)

def test_entity_span_mapping_overrides_token_text():
    # Create Business Partner in SAP.
    sent = "Create Business Partner in SAP."
    state = {"chunks": [{"text": sent, "start": 0, "end": len(sent)}]}
    state = node_pos_dep(state)

    # entity nhiều từ: "Business Partner" (Object) + "SAP" (System)
    bps, bpe = _span(sent, "Business Partner")
    ss, se = _span(sent, "SAP")
    state["entities"] = [
        {"text": "Business Partner", "label": "Object", "start": bps, "end": bpe},
        {"text": "SAP", "label": "System", "start": ss, "end": se},
    ]

    rels = extract_relations_from_state(state)
    # object_of: Create -> Business Partner (ưu tiên entity span)
    assert {"head": "Create", "tail": "Business Partner", "type": "object_of"} in rels
    # exec_in: Create -> SAP
    assert {"head": "Create", "tail": "SAP", "type": "exec_in"} in rels

def test_multi_sentence_no_cross_sequence():
    txt = "Connect to SAP system. Then create a request."
    state = {"chunks": [{"text": txt, "start": 0, "end": len(txt)}]}
    state = node_pos_dep(state)
    state["entities"] = []  # không cần entity

    rels = extract_relations_from_state(state)
    # Có 2 câu, mỗi câu có động từ chính; không nên nối cross-sentence
    # Chỉ cần đảm bảo không có sequence_next từ "Connect" → "create" nếu chúng ở 2 câu khác nhau
    assert {"head": "Connect", "tail": "create", "type": "sequence_next"} not in rels
    # Nhưng vẫn có thể có quan hệ khác trong từng câu (không assert cụ thể ở đây)
