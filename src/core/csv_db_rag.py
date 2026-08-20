import numpy as np
import os
import faiss
from indexing.update_index import main as db_indexing
from dotenv import load_dotenv

load_dotenv()
FILE_PATH = os.environ["FILE_PATH"]

# ⚠️ Ya NO ejecutamos db_indexing() ni leemos el CSV al importar.
# El re-indexado se corre aparte, no al arrancar el bot.

def text2Vec(data, embedding):
    return embedding.embed_query(data)

def search_Index(user_messages, embedding, top_k=5):
    """Search with FAISS index and return top_k results."""
    index_path = f"{FILE_PATH}/data/csv_index/faiss_dump/faiss_index"
    metadata_path = f"{FILE_PATH}/data/csv_index/faiss_dump/faiss_metadata.txt"

    user_query = np.array(text2Vec(user_messages, embedding), dtype=float)
    query_vector = user_query / np.linalg.norm(user_query)  # normaliza (coseno)
    if len(query_vector.shape) == 1:
        query_vector = query_vector.reshape(1, -1)  # FAISS necesita 2D

    index = faiss.read_index(index_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta_data = f.readlines()

    distances, indices = index.search(query_vector, top_k)
    results = [meta_data[indices[0][i]].strip() for i in range(top_k)]
    return results