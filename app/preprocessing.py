# Preprocess with spaCy (English-only) + chunk summarization + caching
import re
import html
import unicodedata
import hashlib
import shelve
import os
from typing import List, Dict, Any, TypedDict, Optional, Callable
from functools import lru_cache

from app.state import PipelineState

# --- spaCy import with helpful error message if missing ---
try:
    import spacy
    from spacy.tokens import Doc, Span
except Exception as e:
    raise ImportError(
        "spaCy is required for this preprocess module. "
        "Install with `pip install spacy` and download the English model: "
        "`python -m spacy download en_core_web_sm`."
    )

# load english model once (small model recommended for speed)
# If you want a different model (en_core_web_trf), change here.
_SPACY_MODEL_NAME = os.environ.get("SPACY_MODEL", "en_core_web_sm")
_nlp = spacy.load(_SPACY_MODEL_NAME, disable=["lemmatizer"])  # keep tagger/ner/parser for sentences & entities


# ----- Utilities -----
RE_Control = re.compile(r'[\r\t\x0b\x0c]')

RE_EMOJI = re.compile(
    "["                     
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\u2600-\u26FF\u2700-\u27BF"
    "]+",
    flags=re.UNICODE
)

def normalize_text(text: str, preserve_case: bool = True, remove_emojis: bool = False) -> str:
    if text is None:
        return ""
    txt = text.strip()
    txt = html.unescape(txt)
    txt = unicodedata.normalize("NFC", txt)
    txt = RE_Control.sub(" ", txt)
    if remove_emojis:
        txt = RE_EMOJI.sub("", txt)
    txt = re.sub(r'\s+', ' ', txt)
    if not preserve_case:
        txt = txt.lower()
    return txt

# ----- Caching -----
# In-memory cache (fast) and optional disk cache via shelve (persistent)
_IN_MEMORY_CACHE: Dict[str, Any] = {}

def _cache_key(text: str, max_len_chars: int, overlap_chars: int, preserve_case: bool, remove_emojis: bool) -> str:
    h = hashlib.sha256()
    h.update(text.encode("utf-8"))
    h.update(f"|max{max_len_chars}|ov{overlap_chars}|pc{int(preserve_case)}|re{int(remove_emojis)}".encode())
    return h.hexdigest()

def _load_from_disk_cache(key: str, cache_path: Optional[str]):
    if not cache_path:
        return None
    try:
        with shelve.open(cache_path) as db:
            return db.get(key)
    except Exception:
        return None

def _save_to_disk_cache(key: str, value: Any, cache_path: Optional[str]):
    if not cache_path:
        return
    try:
        with shelve.open(cache_path) as db:
            db[key] = value
    except Exception:
        pass

# ----- spaCy-based sentence tokenization with offsets and tokenization -----
def sentence_tokenize_with_offsets_spacy(text: str) -> List[Dict[str, Any]]:
    """
    Uses spaCy to split into sentences and returns list of dicts:
    { "text": sent_text, "start": char_start, "end": char_end, "tokens": [...], "ents": [ (ent_text, ent_label) ] }
    """
    doc: Doc = _nlp(text)
    out = []
    for sent in doc.sents:
        sent_text = sent.text.strip()
        if not sent_text:
            continue
        start = sent.start_char
        end = sent.end_char
        tokens = [token.text for token in sent]
        ents = [(ent.text, ent.label_) for ent in sent.ents]
        out.append({
            "text": sent_text,
            "start": start,
            "end": end,
            "tokens": tokens,
            "ents": ents,
            "token_count": len(tokens),
        })
    # fallback: if spaCy didn't find sentences (rare), return whole text as one sentence
    if not out and text.strip():
        tokens = [t.text for t in _nlp(text)]
        out.append({"text": text.strip(), "start": 0, "end": len(text), "tokens": tokens, "ents": [], "token_count": len(tokens)})
    return out

# ----- Chunking with overlap -----

def chunk_sentences(sentences: List[Dict[str, Any]],
                    full_text: str,
                    max_len_chars: int = 1200,
                    overlap_chars: int = 120) -> List[Dict[str, Any]]:
    chunks = []
    cur_start = None
    cur_end = None
    cur_pieces = []
    cur_token_count = 0
    for s in sentences:
        s_len = s["end"] - s["start"]
        if cur_start is None:
            cur_start = s["start"]
            cur_end = s["end"]
            cur_pieces = [s]
            cur_token_count = s.get("token_count", 0)
        else:
            projected_len = (cur_end - cur_start) + (s["end"] - s["start"]) + 1
            if projected_len <= max_len_chars:
                cur_pieces.append(s)
                cur_end = s["end"]
                cur_token_count += s.get("token_count", 0)
            else:
                # finalize chunk
                chunk_text = " ".join([p["text"] for p in cur_pieces]).strip()
                chunks.append({
                    "text": chunk_text,
                    "start": cur_start,
                    "end": cur_end,
                    "sentence_count": len(cur_pieces),
                    "token_count": cur_token_count,
                    "sentences": cur_pieces,
                })
                # start next chunk
                cur_start = s["start"]
                cur_end = s["end"]
                cur_pieces = [s]
                cur_token_count = s.get("token_count", 0)
    # last
    if cur_start is not None:
        chunk_text = " ".join([p["text"] for p in cur_pieces]).strip()
        chunks.append({
            "text": chunk_text,
            "start": cur_start,
            "end": cur_end,
            "sentence_count": len(cur_pieces),
            "token_count": cur_token_count,
            "sentences": cur_pieces,
        })

    # Add overlap context preview (not modifying start/end)
    for i in range(1, len(chunks)):
        prev = chunks[i-1]
        cur = chunks[i]
        # take last `overlap_chars` from prev.text as overlap_preview
        overlap_preview = prev["text"][-overlap_chars:] if overlap_chars > 0 else ""
        cur["overlap_preview"] = overlap_preview
    if chunks:
        chunks[0]["overlap_preview"] = ""
    return chunks

# ----- Simple extractive summarizer (heuristic) -----
def extractive_summary_for_chunk(chunk: Dict[str, Any], max_chars: int = 300, sentences_to_pick: int = 2) -> str:
    """
    Score sentences by:
      - number of named entities (more entities -> higher)
      - sentence length (longer up to a point)
      - earlier position slightly preferred
    Pick top sentences and join in original order until <= max_chars.
    """
    sents = chunk.get("sentences", [])
    if not sents:
        return ""
    scores = []
    for idx, s in enumerate(sents):
        ent_count = len(s.get("ents", []))
        length_score = min(1.0, (s.get("token_count", 0) / 30.0))  # heuristic
        position_score = 1.0 / (1 + idx)
        score = ent_count * 2.0 + length_score * 0.5 + position_score * 0.3
        scores.append((idx, score))
    # choose top-k sentences by score
    scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)
    chosen_idx = sorted([i for i, _ in scores_sorted[:sentences_to_pick]])
    summary_pieces = []
    cur_len = 0
    for i in chosen_idx:
        s_text = sents[i]["text"]
        if cur_len + len(s_text) + 1 > max_chars and summary_pieces:
            break
        summary_pieces.append(s_text)
        cur_len += len(s_text) + 1
    summary = " ".join(summary_pieces)
    # fallback: if summary empty, use preview (truncate)
    if not summary:
        t = chunk.get("text", "")
        summary = t if len(t) <= max_chars else t[:max_chars-3] + "..."
    return summary

# ----- Main node_preprocess with spaCy + caching + summarization -----
def node_preprocess(state: PipelineState = {},
                    max_len_chars: int = 1200,
                    overlap_chars: int = 120,
                    preserve_case: bool = True,
                    remove_emojis: bool = False,
                    use_disk_cache: bool = False,
                    disk_cache_path: Optional[str] = "/tmp/langgraph_preproc_cache.db",
                    summarizer: Optional[Callable[[Dict[str, Any]], str]] = None,
                    summary_max_chars: int = 300) -> PipelineState:
    """
    Main preprocess node.
    - Normalizes text
    - Uses spaCy to split sentences (with offsets) and tokens/ents
    - Chunks sentences into chunks with overlap
    - Summarizes each chunk (extractive) using `summarizer` if provided else default heuristic
    - Caching: in-memory + optional disk shelve cache
    """
    raw = state.get("text", "") or ""
#     raw ="""After connecting to the SAP system, Email and Google Drive, The process get information of purchase from Email and checks if the purchase amount exceeds 5000 USD. If true, it creates a new Business Partner in SAP and sends a confirmation email to the requester. Otherwise, it uploads the purchase document to Google Drive for manual approval and notifies the finance team.
# """ 
    state["text"] = raw
    key = _cache_key(raw, max_len_chars, overlap_chars, preserve_case, remove_emojis)
    # try in-memory cache
    cached = _IN_MEMORY_CACHE.get(key)
    if cached:
        state["preprocess"] = cached["preprocess"]
        state["chunks"] = cached["chunks"]
        state.setdefault("meta", {})
        state["meta"]["cleaned_text"] = cached["cleaned_text"]
        return state
    # try disk cache
    if use_disk_cache:
        disk_val = _load_from_disk_cache(key, disk_cache_path)
        if disk_val:
            _IN_MEMORY_CACHE[key] = disk_val
            state["preprocess"] = disk_val["preprocess"]
            state["chunks"] = disk_val["chunks"]
            state.setdefault("meta", {})
            state["meta"]["cleaned_text"] = disk_val["cleaned_text"]
            return state

    # normalize
    cleaned = normalize_text(raw, preserve_case=preserve_case, remove_emojis=remove_emojis)

    # sentence tokenize with spaCy
    sentences = sentence_tokenize_with_offsets_spacy(cleaned)

    # chunk
    chunks_meta = chunk_sentences(sentences, cleaned, max_len_chars=max_len_chars, overlap_chars=overlap_chars)

    # enrich chunk metadata: tokens, preview, summary
    for c in chunks_meta:
        c_text = c["text"]
        # token sample via spaCy to preserve subtokenization better
        doc = _nlp(c_text)
        tokens = [t.text for t in doc][:200]
        c["tokens_sample"] = tokens[:20]
        c["preview"] = c_text if len(c_text) <= 200 else c_text[:197] + "..."
        # summary
        if summarizer:
            try:
                s = summarizer(c)
            except Exception:
                s = extractive_summary_for_chunk(c, max_chars=summary_max_chars)
        else:
            s = extractive_summary_for_chunk(c, max_chars=summary_max_chars)
        c["summary"] = s
        # keep clean_text as convenience
        c["clean_text"] = c_text

    preprocess_meta = {
        "raw_text_len": len(raw),
        "cleaned_text_len": len(cleaned),
        "n_sentences": len(sentences),
        "n_chunks": len(chunks_meta),
        "max_len_chars": max_len_chars,
        "overlap_chars": overlap_chars,
        "preserve_case": preserve_case,
        "remove_emojis": remove_emojis,
        "spacy_model": _SPACY_MODEL_NAME,
    }

    result_obj = {
        "preprocess": preprocess_meta,
        "chunks": chunks_meta,
        "cleaned_text": cleaned,
    }

    # save caches
    _IN_MEMORY_CACHE[key] = result_obj
    if use_disk_cache:
        _save_to_disk_cache(key, result_obj, disk_cache_path)

    # update state

    state["preprocess"] = preprocess_meta
    state["chunks"] = chunks_meta
    state.setdefault("meta", {})
    state["meta"]["cleaned_text"] = cleaned
    print(state)
    return state

if __name__ == "__main__" :
    node_preprocess()
    