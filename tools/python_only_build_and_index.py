# python_only_build_and_index.py
import re, json, json5, uuid, os, sys
from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

TS_FILE = Path("activity.ts")          # đổi nếu đặt tên khác
OUT_JSON = Path("activity_tpl_docs.json")
CHROMA_DIR = Path("../chroma_activity")
COLLECTION = "activity_pkg"
EMB_MODEL = "intfloat/multilingual-e5-base"

def extract_activity_packages(ts_text: str) -> str:
    """
    Cắt đúng đoạn mảng ActivityPackages từ file TS ở dạng chuỗi JSON5.
    """
    # lấy phần sau 'export const ActivityPackages ='
    m = re.search(r"export\s+const\s+ActivityPackages\s*=\s*(\[)", ts_text)
    if not m:
        raise RuntimeError("Không tìm thấy 'export const ActivityPackages = [' trong activity.ts")

    start = m.start(1)
    # tìm dấu '];' đóng mảng ở cùng cấp
    # cách đơn giản: duyệt đếm ngoặc để tìm vị trí đóng của mảng
    i, depth = start, 0
    while i < len(ts_text):
        c = ts_text[i]
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                # lấy đến ký tự ']' này
                end = i + 1
                break
        i += 1
    else:
        raise RuntimeError("Không tìm được dấu ']' kết thúc mảng ActivityPackages.")

    raw_array = ts_text[start:end]

    # Tiền xử lý vài thứ không phải JSON:
    # 1) Xoá comment // ... và /* ... */
    raw_array = re.sub(r"//.*", "", raw_array)
    raw_array = re.sub(r"/\*.*?\*/", "", raw_array, flags=re.S)

    # 2) Thay các biểu thức TS/enum không hợp lệ JSON:
    #    overrideType: RFVarType["any"] | RFVarType["dictionary"] | RFVarType["list"] | RFVarType["scalar"]
    raw_array = re.sub(r'RFVarType\["any"\]', '"$"', raw_array)
    raw_array = re.sub(r'RFVarType\["dictionary"\]', '"&"', raw_array)
    raw_array = re.sub(r'RFVarType\["list"\]', '"@"', raw_array)
    raw_array = re.sub(r'RFVarType\["scalar"\]', '"$"', raw_array)

    # 3) Một số field có thể là trailing comma — json5 chấp nhận, nên không cần xoá.
    return raw_array

def build_tpl_docs(activity_packages):
    """
    Mirror logic buildTplDocs() trong TS:
    - text = pkg.displayName + t.displayName + t.description + t.keyword + join("k type keywordArg")
    - requiredArgs = map từ t.arguments
    """
    tpl_docs = []
    for pkg in activity_packages:
        pkg_display = (pkg.get("displayName") or "")
        pkg_id = pkg.get("_id") or ""
        templates = pkg.get("activityTemplates") or []
        for t in templates:
            template_id = t.get("templateId") or ""
            keyword = t.get("keyword") or ""
            t_display = t.get("displayName") or ""
            t_desc = t.get("description") or ""
            args = t.get("arguments") or {}

            # text pieces
            pieces = [pkg_display, t_display, t_desc, keyword]
            # map arguments: "key type keywordArg"
            for k, v in args.items():
                if not isinstance(v, dict):
                    continue
                arg_type = v.get("type", "")
                kwarg = v.get("keywordArg", "")
                pieces.append(f"{k} {arg_type} {kwarg}")

            text = " ".join(pieces).strip().lower()

            # requiredArgs
            required_args = []
            for k, v in args.items():
                if not isinstance(v, dict):
                    continue
                required_args.append({
                    "name": k,
                    "type": v.get("type"),
                    "keywordArg": v.get("keywordArg")
                })

            tpl_docs.append({
                "templateId": template_id,
                "pkg": pkg_id,
                "text": text,
                "keyword": keyword,
                "requiredArgs": required_args
            })
    return tpl_docs
def _sanitize_meta(meta: dict) -> dict:
    def to_primitive(x):
        # Cho phép: str, int, float, bool, None
        if isinstance(x, (str, int, float, bool)) or x is None:
            return x
        # Còn lại: list/dict/… -> stringify
        return json.dumps(x, ensure_ascii=False)
    return {k: to_primitive(v) for k, v in meta.items()}
def main():
    if not TS_FILE.exists():
        print(f"Không thấy file {TS_FILE.resolve()}. Đặt script cạnh activity.ts hoặc chỉnh TS_FILE.")
        sys.exit(1)

    ts_text = TS_FILE.read_text(encoding="utf-8")
    raw_array = extract_activity_packages(ts_text)

    # Parse JSON5 → Python
    try:
        activity_packages = json5.loads(raw_array)
    except Exception as e:
        # Gợi ý debug khi fail parse
        print("Lỗi parse JSON5 từ activity.ts. Hãy kiểm tra các biểu thức TS lạ chưa được thay thế.")
        raise

    # Build docs
    docs = build_tpl_docs(activity_packages)

    # Ghi JSON để tái sử dụng nếu cần
    OUT_JSON.write_text(json.dumps(docs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(docs)} templates to {OUT_JSON}")

    # Index vào Chroma
    emb = HuggingFaceEmbeddings(model_name=EMB_MODEL)
    texts = [d["text"] for d in docs]
    metas_raw = [{
        "templateId": d["templateId"],
        "pkg": d["pkg"],
        "keyword": d["keyword"],
        "requiredArgs": d["requiredArgs"],   # <-- list[dict], sẽ bị từ chối nếu không sanitize
    } for d in docs]

    metas = [_sanitize_meta(m) for m in metas_raw]
    ids = [str(uuid.uuid4()) for _ in texts]

    vs = Chroma(collection_name=COLLECTION, persist_directory=str(CHROMA_DIR), embedding_function=emb)
    vs.add_texts(texts=texts, metadatas=metas, ids=ids)
    vs.persist()
    print(f"Indexed {len(texts)} activity templates into {CHROMA_DIR} (collection='{COLLECTION}').")

    # Query: get first 10 rows in vs
   
def query_sample(vs):
    results = vs.get(include=["metadatas", "documents", "embeddings"])
    for i in range(min(5, len(results["documents"]))):
        print(f"Row {i+1}:")
        print("  Text:", results["documents"][i])
        print("  Metadata:", results["metadatas"][i])
        print("  Vector (embedding):", len(results["embeddings"][i]), results["embeddings"][i][0:5], "...")
    vec_pairs = vs.similarity_search_with_score("send email using gmail", k=10)
    for doc, score in vec_pairs:
        print(f"\nSimilarity search result (score={score}):")
        print(" ", doc.metadata)
        print(" ", doc.page_content[:100], "...")

if __name__ == "__main__":
    query_sample(Chroma(collection_name=COLLECTION, persist_directory=str(CHROMA_DIR), embedding_function=HuggingFaceEmbeddings(model_name=EMB_MODEL)))
