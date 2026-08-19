import faiss
import pandas as pd
import numpy as np
import ast
import os
import re

def load_csv_and_prepare_vectors(csv_file, vector_column="Vector", meta_column="Contents"):
    """
    CSVファイルを読み込み、ベクトルデータとメタデータを準備する
    """
    df = pd.read_csv(csv_file)

    # ベクトルデータをリストに変換
    vectors = []
    meta_data = []
    for idx, row in df.iterrows():
        try:
            vec = np.array(ast.literal_eval(row[vector_column]), dtype=np.float32)
            vectors.append(vec)
            meta_data.append(row[meta_column])
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
    print("metadata：", meta_data)
    
    # ベクトルをNumPy配列に変換
    vectors = np.array(vectors)
    return vectors, meta_data

# def build_faiss_index_with_metadata(vectors, meta_data, index_path, metadata_path):
#     """
#     FAISSインデックスを作成し、メタデータを別途保存
#     """
#     d = vectors.shape[1]  # ベクトル次元数

#     # FAISSインデックスを作成
#     index = faiss.IndexFlatL2(d)  # L2距離を使用
#     index.add(vectors)  # ベクトルを追加

#     # インデックスを保存
#     faiss.write_index(index, index_path)
#     print(f"FAISS index saved to {index_path}")

#     # メタデータを保存
#     with open(metadata_path, "w", encoding="utf-8") as f:
#         for meta in meta_data:
#             f.write(f"{meta}\n")
#     print(f"Metadata saved to {metadata_path}")
    
def normalize_vectors(vectors):
    """
    ベクトルを正規化してノルムを1にする
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms

def save_metadata(metadata_list, metadata_path):
    """
    メタデータをテキストファイルとして保存
    """
    with open(metadata_path, "w", encoding="utf-8") as f:
        for metadata in metadata_list:
            # 改行をエスケープして保存
            normalize_metadata = metadata.replace("\n", "").replace("\u3000", "")
            normalize_metadata = re.sub(r'Writer:.*', '', normalize_metadata)
            f.write(f"{normalize_metadata}\n")
    print(f"Metadata saved to {metadata_path}")

def build_faiss_index_with_metadata(vectors, meta_data, index_path, metadata_path):
    """
    コサイン類似度用のFAISSインデックスを作成し、保存する
    """
    # ベクトルを正規化
    normalized_vectors = normalize_vectors(vectors)

    # インデックス作成（L2距離を使用するが、正規化済みベクトルによりコサイン類似度に相当）
    d = normalized_vectors.shape[1]  # 次元数
    index = faiss.IndexFlatIP(d)  # 内積 (Inner Product) を使用
    index.add(normalized_vectors)  # 正規化ベクトルを追加

    # メタデータを保存
    save_metadata(meta_data, metadata_path)
    
    # インデックスを保存
    faiss.write_index(index, index_path)
    print(f"FAISS index saved to {index_path}")