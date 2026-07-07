# 存成 check_index.py，项目根目录跑
import chromadb
from collections import Counter

client = chromadb.PersistentClient(path="data/index")
col = client.get_collection("filings")
got = col.get(include=["metadatas"])

print(f"total {len(got['ids'])} vectors")
print("classifying according to the types:", Counter(m["form"] for m in got["metadatas"]))
print("classifying according to the companys:", Counter(m["ticker"] for m in got["metadatas"]))