import numpy as np
from numpy.linalg import norm
import shutil
import ast
import os
import json
import faiss
import pandas as pd
from indexing.update_index import main as db_indexing
from dotenv import load_dotenv

load_dotenv()

# Initialize environment variables
FILE_PATH = os.environ["FILE_PATH"]

db_indexing()

csv_file = f"{FILE_PATH}/db/csv_index/vector_index.csv"
df = pd.read_csv(csv_file)

def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

def text2Vec(data, embedding):
  vec_data = embedding.embed_query(data)
  return vec_data

# def search_Index(user_messages, embedding):
#     similarities = []
#     user_query = text2Vec(user_messages, embedding)
#     user_query = np.array(user_query, dtype=float)  # user_queryを数値ベクトルに変換

#     for v in df["Vector"]:
#         # 文字列として保存されているベクトルをリストに変換
#         if isinstance(v, str):
#             vec = np.array(ast.literal_eval(v), dtype=float)
#             #vec = vec = np.array(json.loads(v), dtype=float)
#         else:
#             vec = np.array(v, dtype=float)

#         # コサイン類似度を計算
#         sim = cosine_similarity(user_query, vec)
#         similarities.append(sim)

#     # DataFrameに類似度カラムを追加
#     df["similarity"] = similarities

#     # 類似度の高い上位5件を取得
#     top_5 = df.sort_values(by="similarity", ascending=False).head(5)
#     return top_5["Contents"].tolist()

def search_faiss_index_with_metadata(index_path, metadata_path, query_vector, top_k=5):
    """
    FAISSインデックスを読み込み、検索し、対応するメタデータを返す
    """
    # インデックスを読み込み
    index = faiss.read_index(index_path)

    # メタデータを読み込み
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta_data = f.readlines()

    # 検索を実行
    distances, indices = index.search(query_vector, top_k)

    # 検索結果のメタデータを取得
    results = [{"distance": distances[0][i], "metadata": meta_data[indices[0][i]].strip()} for i in range(top_k)]
    return results

# def search_Index(user_messages, embedding):
#     top5 = []
#     user_query = text2Vec(user_messages, embedding)
#     user_query = np.array(user_query, dtype=float)  # user_queryを数値ベクトルに変換
#     faiss_index_path = f"{FILE_PATH}/db/csv_index/faiss_dump/faiss_index"
#     metadata_path = f"{FILE_PATH}/db/csv_index/faiss_dump/faiss_metadata.txt"
#     results = search_faiss_index_with_metadata(faiss_index_path, metadata_path, user_query)

#     # 検索結果を表示
#     for result in results:
#         top5.append(result['metadata'])
#         return top5
def search_Index(user_messages, embedding, top_k=5):
    """
    コサイン類似度を利用して検索し、対応するメタデータを返す
    """
    index_path = f"{FILE_PATH}/db/csv_index/faiss_dump/faiss_index"
    metadata_path = f"{FILE_PATH}/db/csv_index/faiss_dump/faiss_metadata.txt"
    user_query = text2Vec(user_messages, embedding)
    user_query = np.array(user_query, dtype=float)  # user_queryを数値ベクトルに変換
    # クエリベクトルを正規化
    query_vector = user_query / np.linalg.norm(user_query)

    # クエリベクトルを2次元に変換
    if len(query_vector.shape) == 1:
        query_vector = query_vector.reshape(1, -1)

    # インデックスを読み込み
    index = faiss.read_index(index_path)

    # メタデータを読み込み
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta_data = f.readlines()

    # 検索を実行
    distances, indices = index.search(query_vector, top_k)

    # 検索結果のメタデータを取得
    results = [meta_data[indices[0][i]].strip() for i in range(top_k)]
    print(f"results：{results}")
    return results