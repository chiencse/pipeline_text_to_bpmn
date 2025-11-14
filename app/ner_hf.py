# app/ner_hf.py
"""
Utilities for running a fine-tuned Hugging Face NER model in the LangGraph pipeline.

- Đặt biến môi trường HF_NER_MODEL để trỏ tới thư mục local chứa artifact fine-tune
  (config.json, tokenizer.json, pytorch_model.bin, v.v.) hoặc repo id trên Hugging Face Hub.
  Mặc định: "outputs/xlmrb-ner".
- Hàm public: run_hf_ner(chunks: List[str]) -> List[Dict]
  Trả về danh sách entity có schema: {label, text, start, end, score}
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Dict, List, Any, Optional

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

# -----------------------------------------------------------------------------
# Cấu hình
# -----------------------------------------------------------------------------

HF_NER_MODEL = os.environ.get("HF_NER_MODEL", "D:\\BK\\DACN\\ner-pipeline-en-starter\\outputs\\xlmrb-ner")

# Nếu bạn muốn chuẩn hoá nhãn về ACT/SYS/DOC, chỉnh map bên dưới tùy theo nhãn model.
# Ví dụ: model xuất "ACTION", "SYSTEM", "DOCUMENT" hoặc B-*/I-*.
_LABEL_CANON_MAP = {
    "ACT": "ACT",
    "ACTION": "ACT",
    "SYS": "SYS",
    "SYSTEM": "SYS",
    "DOC": "DOC",
    "DOCUMENT": "DOC",
}

def _canon_label(lbl: str) -> str:
    x = (lbl or "").upper()
    # Bỏ prefix BIO
    x = x.replace("B-", "").replace("I-", "")
    return _LABEL_CANON_MAP.get(x, x)


# -----------------------------------------------------------------------------
# Loader pipeline (memoized)
# -----------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_ner_pipeline(model_name_or_path: str = HF_NER_MODEL):
    """
    Load token-classification pipeline một lần (memoized).
    Tự động chọn GPU nếu có.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        model = AutoModelForTokenClassification.from_pretrained(model_name_or_path)
    except OSError as exc:
        raise RuntimeError(
            "Cannot load Hugging Face NER model. "
            "Set HF_NER_MODEL to the directory or Hub repo that contains your fine-tuned artifacts."
        ) from exc

    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        task="token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",  # gộp subword thành span
        device=device,
    )


# -----------------------------------------------------------------------------
# Chuẩn hoá output
# -----------------------------------------------------------------------------

def _normalize_entity(raw: Dict[str, Any], base_offset: int) -> Optional[Dict[str, Any]]:
    """
    Convert pipeline outputs về schema dùng trong pipeline:
    {label, text, start, end, score}
    start/end là offset tuyệt đối (sau khi ghép các chunk).
    """
    # Hỗ trợ cả key "entity_group" (HF >= 4.10) và "entity" (tuỳ model)
    label = raw.get("entity_group") or raw.get("entity")
    text = raw.get("word") or raw.get("text")
    start = raw.get("start")
    end = raw.get("end")
    score = raw.get("score", 0.0)

    if label is None or text is None or start is None or end is None:
        # Bỏ các span không đủ trường (bảo toàn pipeline)
        return None

    return {
        "label": _canon_label(str(label)),
        "text": str(text),
        "start": int(start) + base_offset,
        "end": int(end) + base_offset,
        "score": float(score),
    }



def run_hf_ner(chunks: List[str]) -> List[Dict[str, Any]]:
    """
    Chạy NER trên danh sách text chunks và trả về danh sách entity chuẩn hoá.

    Parameters
    ----------
    chunks : List[str]
        Danh sách chuỗi sau bước tiền xử lý (simple_chunk).

    Returns
    -------
    List[Dict[str, Any]]
        Mỗi entity gồm: {label, text, start, end, score}
        - label: đã được chuẩn hoá theo _LABEL_CANON_MAP (ACT/SYS/DOC nếu cấu hình)
        - start/end: offset tuyệt đối tính theo tổng văn bản ghép các chunk
        - score: độ tin cậy từ model
    """
    print("Running HF NER on chunks:", chunks)
    if not isinstance(chunks, list):
       
        return []

    ner = _load_ner_pipeline()
    entities: List[Dict[str, Any]] = []
    offset = 0

    for chunk in chunks:
        sents = chunk.get("sentences", "")
        for sent in sents:
            text = sent.get("text", "")
            if not text:
                offset += 1  # khớp quy ước nối chuỗi (ví dụ 1 ký tự phân cách)
                continue
            try:
                preds = ner(text, truncation=True)
            except TypeError:
                preds = ner(text)
            # truncation=True giúp tránh lỗi nếu chunk vượt max_length tokenizer
        
            if isinstance(preds, list):
                for raw_ent in preds:
                    norm = _normalize_entity(raw_ent, offset)
                    if norm is not None:
                        entities.append(norm)

            # Giữ đúng quy ước offset với cách bạn chunk/join ở preprocess.
            # Nếu simple_chunk của bạn chèn '\n\n' khi ghép, hãy đổi +1 -> +2.
            offset += len(text) + 1

    return entities


__all__ = ["run_hf_ner"]
if __name__ == "__main__":
    # Test nhanh
    os.environ.setdefault("HF_NER_MODEL", "D:\\BK\\DACN\\ner-pipeline-en-starter\\outputs\\xlmrb-ner")
    sample_chunks = [
        "Create Sales Order in SAP and then send quotation to the customer.", "If total amount > 10000, request manager approval; otherwise auto-approve.", "Invoice INV-2024-0157 must reference PO-98912 and be posted by Accountant."

    ]
    ents = run_hf_ner(sample_chunks)
    print("\nDetected entities:")
    for e in ents:
        print(e)
