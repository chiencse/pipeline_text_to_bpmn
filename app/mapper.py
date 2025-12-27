from typing import List, Dict, Tuple

def mix_score(bm25: float|None, cosine_sim: float|None, rule_bonus: float=0.0) -> float:
    b = bm25 if bm25 is not None else 0.0
    c = cosine_sim if cosine_sim is not None else 0.0
    return 0.5*b + 0.4*c + 0.1*rule_bonus

def decide_mapping(score: float) -> Tuple[str, bool]:
    if score >= 0.75:
        return ("ServiceTask", True)   # map chắc
    # if score >= 0.60:
    #     return ("AdapterTask", True)    # cần review
    return ("ManualTask", False)        # không map

def post_map_task(task: Dict, candidates: List[Dict], rule_bonus: float = 0.0) -> Dict:
    """
    task = {"id","name","type","lane","pool"}
    candidates = [{"activity_id","score","requiredArgs":[{name,type,keywordArg}], ...}, ...]
    """
    if not candidates:
        t, rev = decide_mapping(0.0)
        return {
            "node_id": task["id"], "activity_id": None, "confidence": None,
            "manual_review": (t == "AdapterTask"),
            "type": t, "candidates": []
        }
    best = max(candidates, key=lambda x: x["score"])
    t, rev = decide_mapping(best["score"])
    rec = {
        "node_id": task["id"],
        "activity_id": best["activity_id"] ,
        "confidence": best["score"],
        "automatic": rev,
        "type": t,
        "candidates": [{"activity_id": c["activity_id"], "score": c["score"]} for c in candidates[:3]],
        "input_bindings": {},
        "outputs": []
    }
    return rec

def auto_fill_bindings(mapping: Dict, required_args: List[Dict], entities: List[Dict]) -> Dict:
    """
    naive binding: map theo 'keywordArg' ~ entity text keywords
    """
    bindings = {}
    for arg in required_args:
        key = arg.get("keywordArg") or arg.get("name")
        if not key:
            continue
        # tìm entity liên quan theo label heuristic
        val = None
        if "email" in key.lower():
            for e in entities:
                if e["label"] == "ROLE" and "@" in e["text"]:
                    val = e["text"]; break
        if val is None:
            # fallback: tìm DOC/SYS/ID/ROLE theo từ khóa gợi ý
            for e in entities:
                if e["label"] in ("DOC","SYS","ID","ROLE") and key.split("_")[0].lower() in e["text"].lower():
                    val = e["text"]; break
        if val is not None:
            bindings[key] = val
    mapping["input_bindings"] = {**mapping.get("input_bindings", {}), **bindings}
    return mapping
