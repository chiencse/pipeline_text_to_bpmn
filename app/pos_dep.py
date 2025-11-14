import os
import spacy
from spacy import displacy
from typing import Dict, Any, List

# load model once
try:
    _nlp = spacy.load("en_core_web_sm")
except Exception as e:
    raise ImportError("Please install spaCy and the English model: pip install spacy && python -m spacy download en_core_web_sm")

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_dependency_svgs(state: Dict[str, Any],
                         out_dir: str = "out/dep_vis",
                         per_sentence: bool = False,
                         svg_prefix: str = "chunk"):
    """
    Save dependency visualizations as SVG files for each chunk (or each sentence if per_sentence=True).
    Files named as {svg_prefix}_{chunk_idx}.svg  or {svg_prefix}_{chunk_idx}_sent_{sent_idx}.svg
    """
    chunks = state.get("chunks", [])
    if not chunks:
        print("⚠️ save_dependency_svgs: no chunks found in state.")
        return []

    _ensure_dir(out_dir)
    written_files = []
    for ci, chunk in enumerate(chunks):
        text = chunk.get("text", "")
        if not text:
            continue
        doc = _nlp(text)
        if per_sentence:
            for si, sent in enumerate(doc.sents):
                svg = displacy.render(sent, style="dep")
                fname = f"{svg_prefix}_{ci}_sent_{si}.svg"
                fpath = os.path.join(out_dir, fname)
                with open(fpath, "w", encoding="utf-8") as f:
                    f.write(svg)
                written_files.append(fpath)
        else:
            # whole-chunk SVG
            svg = displacy.render(doc, style="dep")
            fname = f"{svg_prefix}_{ci}.svg"
            fpath = os.path.join(out_dir, fname)
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(svg)
            written_files.append(fpath)

    print(f"✅ Saved {len(written_files)} dependency SVG(s) to {out_dir}")
    return written_files


def serve_dependency_for_chunk(state: Dict[str, Any], chunk_index: int = 0, port: int = 5000):
    """
    Serve an interactive dependency visualization in the browser for a specific chunk.
    Note: displacy.serve is blocking (starts a local server). Use in dev environment.
    """
    chunks = state.get("chunks", [])
    if not chunks:
        print("⚠️ serve_dependency_for_chunk: no chunks found.")
        return
    if chunk_index < 0 or chunk_index >= len(chunks):
        print(f"⚠️ serve_dependency_for_chunk: chunk_index {chunk_index} out of range (0..{len(chunks)-1}).")
        return
    text = chunks[chunk_index].get("text", "")
    doc = _nlp(text)
    print(f"Serving dependency visualization for chunk {chunk_index} on http://localhost:{port} ...")
    # This will block until server killed. Good for local dev.
    displacy.serve(doc, style="dep", port=port)


# ---------- Updated node_pos_dep (integrates optional SVG saving) ----------
def node_pos_dep(state: Dict[str, Any] = {},
                 save_svgs: bool = False,
                 svg_out_dir: str = "out/dep_vis",
                 per_sentence_svgs: bool = False) -> Dict[str, Any]:
    """
    Enrich state with POS tags and dependency parse (spaCy).
    Optionally save dependency SVGs for visual inspection.
    - If state['config'] contains keys, they override function args:
        state['config'].get('save_svgs'), state['config'].get('svg_out_dir'), state['config'].get('per_sentence_svgs')
    """
    # allow config override
    cfg = state.get("config", {}) or {}
    save_svgs = cfg.get("save_svgs", save_svgs)
    svg_out_dir = cfg.get("svg_out_dir", svg_out_dir)
    per_sentence_svgs = cfg.get("per_sentence_svgs", per_sentence_svgs)

    chunks = state.get("chunks", [])
    cleaned_text = state.get("meta", {}).get("cleaned_text", None)

    # If no chunks, process whole cleaned_text as single chunk
    if not chunks and cleaned_text:
        chunks = [{"text": cleaned_text, "start": 0, "end": len(cleaned_text), "sentences": [{"text": cleaned_text, "start": 0, "end": len(cleaned_text)}]}]
        state["chunks"] = chunks

    syntax_summary = {"n_chunks": len(chunks), "chunks": []}
    for c_idx, chunk in enumerate(chunks):
        text = chunk.get("text", "")
        if not text:
            continue
        doc = _nlp(text)
        sent_entries = []
        sent_trees = []
        for sent in doc.sents:
            sent_dict = {"text": sent.text, "start": sent.start_char, "end": sent.end_char}
            token_list = []
            for token in sent:
                token_info = {
                    "text": token.text,
                    "idx": token.i,
                    "start": token.idx,
                    "end": token.idx + len(token.text),
                    "pos": token.pos_,
                    "tag": token.tag_,
                    "dep": token.dep_,
                    "head_idx": token.head.i,
                    "head_text": token.head.text,
                    "lemma": token.lemma_,
                    "is_stop": token.is_stop,
                }
                token_list.append(token_info)
            sent_dict["tokens"] = token_list
            tree_lines = [f"{t['text']} ({t['dep']} -> {t['head_text']})" for t in token_list]
            sent_dict["dep_tree_text"] = " | ".join(tree_lines)
            sent_entries.append(sent_dict)
            sent_trees.append(sent_dict["dep_tree_text"])

        # merge with existing sentences if they exist (match by text)
        existing_sents = chunk.get("sentences", [])
        new_sents = []
        for sent in sent_entries:
            matched = False
            for es in existing_sents:
                if es.get("text", "").strip() == sent["text"].strip():
                    es.update(sent)
                    new_sents.append(es)
                    matched = True
                    break
            if not matched:
                new_sents.append(sent)
        chunk["sentences"] = new_sents

        syntax_summary["chunks"].append({
            "chunk_index": c_idx,
            "start": chunk.get("start"),
            "end": chunk.get("end"),
            "dep_trees": sent_trees,
            "n_sentences": len(new_sents)
        })

    state["syntax"] = syntax_summary

    # Optionally save dependency SVGs (for all chunks)
    if save_svgs:
        saved = save_dependency_svgs(state, out_dir=svg_out_dir, per_sentence=per_sentence_svgs, svg_prefix="chunk")
        # store info in state for downstream
        state.setdefault("meta", {})
        state["meta"]["dep_svgs"] = saved

    return state

if __name__ == "__main__":
    # for local testing
    state = {"chunks": [{"text": "An officer selects a menu to accept a new member registration request", "start": 0, "end": 62}, 
                        {"text": "And he will create and send email to the client.", "start": 63, "end": 102    }] }
    state = node_pos_dep(state, save_svgs=True, per_sentence_svgs=True)
    print(state)