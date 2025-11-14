from typing import List, Dict, Any, Optional

def _find_entity_for_span(entities: List[Dict[str, Any]], start: int, end: int) -> Optional[Dict[str, Any]]:
    """Return the first entity that covers the span (or None)."""
    for e in entities or []:
        es, ee = e.get("start"), e.get("end")
        if es is None or ee is None:
            continue
        if es <= start <= ee or es <= end <= ee or (start <= es and ee <= end):
            return e
    return None

def _find_entity_by_text(entities: List[Dict[str, Any]], text: str) -> Optional[Dict[str, Any]]:
    """Fallback: match by substring (case-insensitive)."""
    if not text:
        return None
    t = text.strip().lower()
    for e in entities or []:
        if not e.get("text"):
            continue
        if e["text"].strip().lower() == t:
            return e
    for e in entities or []:
        if t in (e.get("text","").strip().lower()):
            return e
    return None

def extract_relations_from_state(state: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Extract relations using node_pos_dep output stored in state['syntax'] and entities in state['entities'].
    Returns list of relations as dicts: {"head": "...", "tail":"...", "type":"..."}.
    Uses patterns:
      - verb . nsubj -> agent_of (head = verb_text, tail = subject_text/entity)
      - verb . (dobj/obj) -> object_of (head = verb_text, tail = object_text/entity)
      - verb . prep(in) -> exec_in (head = verb_text, tail = pobj_text/entity)
      - sequence_next: verbs order in sentence
    """
    syntax = state.get("syntax", {}) or {}
    chunks = syntax.get("chunks", []) or []
    entities = state.get("entities", []) or []

    rels = []
    for c in chunks:
        # c may include dep_trees (text) and we also have detailed tokens in state['chunks'] sentences
        # Prefer token-level info from state['chunks'] if present (they were enriched in node_pos_dep)
        chunk_idx = c.get("chunk_index")
        # find corresponding chunk in state['chunks'] to get tokens
        chunk_obj = None
        for ch in state.get("chunks", []) or []:
            if ch.get("start") == c.get("start") and ch.get("end") == c.get("end"):
                chunk_obj = ch
                break
        # fallback: try by index
        if chunk_obj is None and isinstance(chunk_idx, int):
            local_chunks = state.get("chunks", []) or []
            if 0 <= chunk_idx < len(local_chunks):
                chunk_obj = local_chunks[chunk_idx]

        # iterate sentences - prefer token lists from chunk_obj['sentences']
        sents = []
        if chunk_obj and chunk_obj.get("sentences"):
            sents = chunk_obj.get("sentences")
        else:
            # fallback to the dep_trees text list
            # c.get("dep_trees") contains lines per sentence -> create minimal items
            for line in c.get("dep_trees", []) or []:
                sents.append({"text": line, "tokens": []})

        for sent in sents:
            tokens = sent.get("tokens", []) or []
            # find verbs in token list (POS tag from node_pos_dep)
            verbs = [t for t in tokens if (t.get("pos") or "").upper() in ("VERB", "AUX")]
            # map token index -> token object
            idx_map = {t.get("idx"): t for t in tokens if t.get("idx") is not None}

            # build nsubj/dobj/prep relations using deps available in token entries
            for v in verbs:
                v_text = v.get("text")
                v_idx = v.get("idx")
                # children relations: tokens with head_idx == v_idx
                children = [t for t in tokens if t.get("head_idx") == v_idx]
                # nsubj
                for c_tok in children:
                    dep = c_tok.get("dep")
                    if dep in ("nsubj", "nsubjpass"):
                        # head: verb, tail: subject (prefer entity span)
                        ent = _find_entity_for_span(entities, c_tok.get("start",0), c_tok.get("end",0)) \
                              or _find_entity_by_text(entities, c_tok.get("text"))
                        tail = ent.get("text") if ent else c_tok.get("text")
                        rels.append({"head": v_text, "tail": tail, "type": "agent_of"})
                # dobj/obj
                for c_tok in children:
                    dep = c_tok.get("dep")
                    if dep in ("dobj", "obj"):
                        ent = _find_entity_for_span(entities, c_tok.get("start",0), c_tok.get("end",0)) \
                              or _find_entity_by_text(entities, c_tok.get("text"))
                        tail = ent.get("text") if ent else c_tok.get("text")
                        rels.append({"head": v_text, "tail": tail, "type": "object_of"})
                # prep -> exec_in (looking for 'in' prep)
                for c_tok in children:
                    if c_tok.get("dep") == "prep" and (c_tok.get("text") or "").lower() == "in":
                        # find pobj under this prep (head_idx == prep.idx)
                        pobj = [t for t in tokens if t.get("head_idx") == c_tok.get("idx") and t.get("dep") == "pobj"]
                        for p in pobj:
                            ent = _find_entity_for_span(entities, p.get("start",0), p.get("end",0)) \
                                  or _find_entity_by_text(entities, p.get("text"))
                            tail = ent.get("text") if ent else p.get("text")
                            rels.append({"head": v_text, "tail": tail, "type": "exec_in"})

            # sequence_next: order verbs by token index
            if verbs:
                verbs_sorted = sorted(verbs, key=lambda t: t.get("idx", 0))
                for i in range(len(verbs_sorted)-1):
                    rels.append({"head": verbs_sorted[i].get("text"), "tail": verbs_sorted[i+1].get("text"), "type": "sequence_next"})

    # deduplicate relations
    uniq = []
    seen = set()
    for r in rels:
        key = (r.get("head"), r.get("tail"), r.get("type"))
        if key not in seen:
            seen.add(key)
            uniq.append(r)
    state["relations"] = uniq
    return state