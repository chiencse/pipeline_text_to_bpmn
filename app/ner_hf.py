from typing import List, Dict

# Stub: Bạn thay bằng loader model HF đã fine-tuned (pipeline token-classification)
# Ở đây trả demo để chạy end-to-end.

def run_hf_ner(chunks: List[str]) -> List[Dict]:
    ents = []
    offset = 0
    for ch in chunks:
        # Demo: đoán đơn giản theo từ
        for m in ("create","send","upload","download","share","delete","move","get"):
            p = ch.lower().find(m)
            if p >= 0:
                ents.append({"label":"ACT","text":m, "start":offset+p,"end":offset+p+len(m)})
        for k in ("Odoo","SAP","Google Drive","Gmail","Google Sheet","Google Classroom","Drive"):
            p = ch.find(k)
            if p >= 0:
                ents.append({"label":"SYS","text":k, "start":offset+p,"end":offset+p+len(k)})
        for d in ("quotation","invoice","file","folder","course","sheet"):
            p = ch.lower().find(d)
            if p >= 0:
                ents.append({"label":"DOC","text":d, "start":offset+p,"end":offset+p+len(d)})
        offset += len(ch) + 1
    return ents
