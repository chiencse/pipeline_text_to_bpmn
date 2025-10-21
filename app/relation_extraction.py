import spacy
from typing import List, Dict

# Load model English nhỏ (đã cài en_core_web_sm trong requirements)
_nlp = spacy.load("en_core_web_sm")

def extract_relations_spacy(text: str, entities: List[Dict]) -> List[Dict]:
    """
    Nếu đã có spans start/end thì có thể map token theo span.
    Ở đây demo pattern nsubj/dobj/prep_in đơn giản.
    """
    doc = _nlp(text)
    rels = []
    # Build quick maps
    verbs = [t for t in doc if t.pos_ == "VERB" or t.pos_ == "AUX"]
    for token in verbs:
        # nsubj -> agent_of
        subs = [c for c in token.children if c.dep_ in ("nsubj","nsubjpass")]
        dobj = [c for c in token.children if c.dep_ in ("dobj","obj")]
        # prep_in -> exec_in
        preps = [c for c in token.children if c.dep_=="prep" and c.text.lower()=="in"]
        for s in subs:
            rels.append({"head": token.text, "tail": s.text, "type": "agent_of"})
        for o in dobj:
            rels.append({"head": token.text, "tail": o.text, "type": "object_of"})
        for p in preps:
            pobj = [c for c in p.children if c.dep_=="pobj"]
            for n in pobj:
                rels.append({"head": token.text, "tail": n.text, "type": "exec_in"})
    # sequence_next naive: theo thứ tự động từ
    for i in range(len(verbs)-1):
        rels.append({"head": verbs[i].text, "tail": verbs[i+1].text, "type": "sequence_next"})
    # unique
    uniq = []
    seen = set()
    for r in rels:
        k = (r["head"], r["tail"], r["type"])
        if k not in seen:
            seen.add(k); uniq.append(r)
    return uniq
