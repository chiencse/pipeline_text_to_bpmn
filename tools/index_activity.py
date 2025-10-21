# tools/index_activity.py
import json, uuid
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

docs = json.load(open("activity_tpl_docs.json", "r", encoding="utf-8"))
emb = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")

texts = [d["text"] for d in docs]
metas = [{"templateId": d["templateId"], "pkg": d["pkg"], "keyword": d["keyword"],
          "requiredArgs": d["requiredArgs"]} for d in docs]

vs = Chroma(collection_name="activity_pkg", persist_directory="./chroma_activity", embedding_function=emb)
ids = [str(uuid.uuid4()) for _ in texts]
vs.add_texts(texts=texts, metadatas=metas, ids=ids)
vs.persist()
print("Indexed", len(texts), "activity templates.")
