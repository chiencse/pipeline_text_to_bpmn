# app/retrieval.py
from __future__ import annotations
from typing import List, Dict, Tuple, Optional
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import numpy as np
import re
from functools import lru_cache

# ---- Vector store (giữ nguyên)
emb = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")
VS = Chroma(collection_name="activity_pkg",
            persist_directory="./chroma_activity",
            embedding_function=emb)

# ---- BM25 setup (in-memory)
# pip install rank-bm25
from rank_bm25 import BM25Okapi

_WORD_RE = re.compile(r"[a-zA-Z0-9_]+")

def _tokenize(text: str) -> List[str]:
    # tokenizer đơn giản: chữ/số/underscore, lower
    return _WORD_RE.findall(text.lower())

@lru_cache(maxsize=1)
def _bm25_index():
    """
    Build BM25 index từ toàn bộ documents trong Chroma (persist).
    Trả về: (bm25, docs, metas)
      - docs: List[str] page_content
      - metas: List[Dict] metadata
    Lưu ý: Chroma trả về dict, không phải tuple.
    """
    res = VS._collection.get(include=["metadatas", "documents"])
    docs = res.get("documents", []) or []
    metas = res.get("metadatas", []) or []

    # Phòng trường hợp độ dài lệch (hiếm)
    n = min(len(docs), len(metas))
    docs = docs[:n]
    metas = metas[:n]

    tokenized_corpus = [_tokenize(d) for d in docs]
    if not tokenized_corpus:
        # Trả về BM25 rỗng an toàn
        class _EmptyBM25:
            def get_scores(self, q_tokens): return np.zeros((0,), dtype=float)
        return _EmptyBM25(), docs, metas

    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, docs, metas

def _normalize(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize an array; nếu hằng số thì trả 0."""
    if arr.size == 0:
        return arr
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-12:
        return np.zeros_like(arr, dtype=float)
    return (arr - lo) / (hi - lo)

def _rule_bonus(query: str, meta: Dict) -> float:
    """
    Bonus đơn giản dựa trên metadata activityTemplate:
      + match keyword chính xác
      + có nhắc tới pkg (vd: google drive / gmail / sheets)
      + trùng tên arg (kéo nhẹ)
    Tổng bonus nằm ~[0, 0.3]
    """
    q = query.lower()
    bonus = 0.0

    # 1) keyword exact / substring
    kw = (meta.get("keyword") or "").lower()
    if kw and kw in q:
        bonus += 0.12

    # 2) package hint
    pkg = (meta.get("pkg") or "").lower()
    # map nhanh vài alias phổ biến
    aliases = {
        "google_drive": ["google drive", "drive", "gdrive"],
        "gmail": ["gmail", "email", "mail"],
        "google_sheets": ["sheets", "google sheet", "spreadsheet"],
        "google_classroom": ["classroom", "gclassroom"],
        "google_form": ["google form", "form"],
        "browser_automation": ["browser", "chrome", "web", "playwright"],
        "file_storage": ["storage", "file storage"],
        "document_automation": ["document", "ocr", "pdf"],
        "rpa-sap-mock": ["sap", "business partner", "s4hana"]
    }
    for k, hints in aliases.items():
        if pkg == k and any(h in q for h in hints):
            bonus += 0.1
            break

    # 3) required args overlap
    arg_names = []
    ra = meta.get("requiredArgs") or []
    # requiredArgs có thể đã bị stringify -> cố parse
    if isinstance(ra, str):
        import json
        try:
            ra = json.loads(ra)
        except Exception:
            ra = []
    for a in ra:
        name = (a.get("name") or "").lower()
        if name and name in q:
            arg_names.append(name)
    # giới hạn tối đa 2 arg để tránh bơm quá tay
    bonus += min(0.05 * len(set(arg_names)), 0.10)

    return min(bonus, 0.30)

def hybrid_search(query: str, k: int = 5,
                  w_bm25: float = 0.4,
                  w_vec: float = 0.5,
                  w_rule: float = 0.1) -> List[Dict]:
    """
    Hybrid search:
      score = w_bm25 * bm25_norm + w_vec * cos_sim_norm + w_rule * rule_bonus
    Trả về top-k ứng viên (đã sort).
    """
    print(f"🔍 Hybrid search for query: '{query}' (k={k})")
    if not query.strip():
        return []

    # --- Vector part (lấy topN trước để tiết kiệm), sau đó union với topN BM25
    topN = max(k * 5, 30)
    vec_pairs = VS.similarity_search_with_score(query, k=topN)
    # cos distance -> similarity ~ 1/(1+d)
    vec_ids = set()
    vec_scores = {}
    vec_docs_map = {}
    for doc, cosdist in vec_pairs:
        sim = 1.0 / (1.0 + float(cosdist))
        tid = doc.metadata.get("templateId")
        if not tid:
            # dùng id nội bộ nếu thiếu
            tid = f"__auto_{id(doc)}"
        vec_ids.add(tid)
        vec_scores[tid] = sim
        vec_docs_map[tid] = (doc.page_content, doc.metadata)

    # --- BM25 part (tính cho toàn bộ, rồi lấy topN)
    bm25, all_docs, all_metas = _bm25_index()
    q_tokens = _tokenize(query)
    all_bm25_scores = np.array(bm25.get_scores(q_tokens), dtype=float)

    # chọn topN theo BM25
    if all_bm25_scores.size:
        top_idx = np.argpartition(-all_bm25_scores, min(topN, all_bm25_scores.size - 1))[:topN]
    else:
        top_idx = np.array([], dtype=int)

    bm25_ids = set()
    bm25_scores = {}
    bm25_docs_map = {}
    for i in top_idx:
        meta = all_metas[i] or {}
        tid = meta.get("templateId") or f"__auto_bm25_{i}"
        bm25_ids.add(tid)
        bm25_scores[tid] = float(all_bm25_scores[i])
        bm25_docs_map[tid] = (all_docs[i], meta)

    # --- Hợp nhất candidate set
    candidate_ids = list(vec_ids.union(bm25_ids))
    if not candidate_ids:
        return []

    # --- Chuẩn hoá thang điểm
    # Vector
    vec_arr = np.array([vec_scores.get(cid, 0.0) for cid in candidate_ids], dtype=float)
    vec_norm = _normalize(vec_arr)

    # BM25
    bm_arr = np.array([bm25_scores.get(cid, 0.0) for cid in candidate_ids], dtype=float)
    bm_norm = _normalize(bm_arr)

    # Rule bonus (không cần normalize; đã giới hạn 0..0.3)
    bonuses = []
    for cid in candidate_ids:
        # Lấy meta/text từ nguồn có sẵn
        if cid in vec_docs_map:
            text, meta = vec_docs_map[cid]
        elif cid in bm25_docs_map:
            text, meta = bm25_docs_map[cid]
        else:
            # fallback: fetch từ collection
            # (ít khi xảy ra)
            text, meta = "", {"templateId": cid}
        bonuses.append(_rule_bonus(query, meta))
    bonuses = np.array(bonuses, dtype=float)

    final_scores = w_bm25 * bm_norm + w_vec * vec_norm + w_rule * bonuses

    # --- Tạo output
    out = []
    for cid, s, v_s, b_s in zip(candidate_ids, final_scores, vec_norm, bonuses):
        # lấy text/meta cuối
        if cid in vec_docs_map:
            text, meta = vec_docs_map[cid]
        else:
            text, meta = bm25_docs_map.get(cid, ("", {}))

        out.append({
            "activity_id": meta.get("templateId"),
            "pkg": meta.get("pkg"),
            "keyword": meta.get("keyword"),
            "requiredArgs": meta.get("requiredArgs", []),
            "score": float(s),
            "bm25_norm": float(bm_norm[candidate_ids.index(cid)]),
            "vec_norm": float(v_s),
            "rule_bonus": float(b_s),
            "text": (text or "")[:240]
        })

    out.sort(key=lambda x: x["score"], reverse=True)

    # --- Print table of candidates with details (with color)
    def color_text(text, color_code):
        return f"\033[{color_code}m{text}\033[0m"

    header = f"{'Rank':<5} {'ActivityID':<20} {'Score':<8} {'BM25':<8} {'Embed':<8} {'Bonus':<8} {'Keyword':<15} {'Pkg':<15}"
    print(color_text(header, "1;36"))  # Bold cyan
    print(color_text("-" * 90, "1;34"))  # Bold blue

    for idx, r in enumerate(out[:k]):
        # Highlight top 1 in green, others in yellow
        if idx == 0:
            row_color = "1;32"  # Bold green
        else:
            row_color = "1;33"  # Bold yellow
        row = f"{idx+1:<5} {str(r['activity_id']):<20} {r['score']:<8.4f} {r['bm25_norm']:<8.4f} {r['vec_norm']:<8.4f} {r['rule_bonus']:<8.4f} {str(r['keyword']):<15} {str(r['pkg']):<15}"
        print(color_text(row, row_color))

    return out[:k]

# def build_query_from_entities(entities: List[Dict]) -> str:
#     """
#     Build a compact query string from extracted entities (preserve order + action-priority).
#     - Uses 'start' if present to determine original order; otherwise input order.
#     - Moves Actions to the front but preserves relative ordering within each label group.
#     - Keeps max_tokens to avoid overly long queries.
#     """
#     if not entities:
#         return "process task flow"

#     # normalize and keep original index
#     ents = []
#     for i, e in enumerate(entities):
#         txt = (e.get("text") or "").strip()
#         if not txt:
#             continue
#         lbl = (e.get("label") or "").strip().lower()
#         start = e.get("start")
#         ents.append({"text": txt, "label": lbl, "start": start, "idx": i})

#     if not ents:
#         return "process task flow"

#     # sort by start if any start exists; else by input order
#     if any(e.get("start") is not None for e in ents):
#         ents_sorted = sorted(ents, key=lambda x: (x.get("start") or 0, x["idx"]))
#     else:
#         ents_sorted = sorted(ents, key=lambda x: x["idx"])

#     # preserve relative order when grouping: collect lists in the order they appear
#     groups = {"action": [], "object": [], "resource": [], "process": [],
#               "system": [], "role": [], "event": [], "condition": [],
#               "datafield": [], "output": []}

#     for e in ents_sorted:
#         lbl = e["label"]
#         if lbl in groups:
#             groups[lbl].append(e["text"])
#         else:
#             # map close variants to sensible buckets
#             if "data" in lbl:
#                 groups["datafield"].append(e["text"])
#             elif lbl in ("doc", "document"):
#                 groups["resource"].append(e["text"])
#             else:
#                 groups["object"].append(e["text"])

#     # create weighted list: actions first (in original relative order), then objects/resources/process, then rest
#     weighted_order = []
#     weighted_order.extend(groups["action"])
#     for k in ("object", "resource", "process"):
#         weighted_order.extend(groups[k])
#     for k in ("system", "role", "event", "condition", "datafield", "output"):
#         weighted_order.extend(groups[k])

#     # remove duplicates while preserving order
#     seen = set()
#     ordered_terms = []
#     for t in weighted_order:
#         if not t or t in seen:
#             continue
#         seen.add(t)
#         ordered_terms.append(t)

#     # limit tokens to avoid too-long queries
#     max_tokens = 40
#     q = " ".join(ordered_terms)
#     toks = q.split()
#     if len(toks) > max_tokens:
#         q = " ".join(toks[:max_tokens])

#     return q or "process task flow"


# if __name__ == "__main__":
#     pass
#     # test nhanh
#     # q = "Send email with Gmail after creating sheet."
#     # res = hybrid_search(q, k=5)
#     # for r in res:
#         # print(f"- {r['activity_id']} (score={r['score']:.4f}): {r['text']}")