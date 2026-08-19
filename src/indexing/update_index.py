from utils.check_file import check_file_exists
import os
import re
import json
import shutil
import ast
import pandas as pd
from langchain.vectorstores.faiss import FAISS
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import AzureOpenAI
import numpy as np
from numpy.linalg import norm
from dotenv import load_dotenv
from indexing.build_faiss import load_csv_and_prepare_vectors, build_faiss_index_with_metadata

load_dotenv()

# Initialize environment variables
FILE_PATH = os.environ["FILE_PATH"]
AZURE_OPENAI_API_KEY = os.environ["AZURE_OPENAI_API_KEY"]
AZURE_OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
LLM_MODELS = os.environ["LLM_MODELS"]
EM_MODELS = os.environ["EM_MODELS"]
MODE = os.environ["MODE"]

llm = AzureChatOpenAI(
    openai_api_version="2024-08-01-preview", #"2024-02-01",
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    openai_api_type="azure",
    openai_api_key=AZURE_OPENAI_API_KEY,
    deployment_name=LLM_MODELS,
    temperature=0,
    streaming=True  # Enable streaming
)

# Initialize embedding model
embedding = AzureOpenAIEmbeddings(
    openai_api_version="2023-05-15",
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    openai_api_type="azure",
    openai_api_key=AZURE_OPENAI_API_KEY,
    model=EM_MODELS,
)

#各データフレームの処理
## データの正規化
def parse_page_content(content: str):
    # id抽出
    id_match = re.search(r'^id:\s*(\d+)', content, flags=re.MULTILINE)
    doc_id = id_match.group(1) if id_match else None
    # Title抽出
    title_match = re.search(r'^Title:\s*(.*)', content, flags=re.MULTILINE)
    title = title_match.group(1).strip() if title_match else None
    # Contents抽出
    # Contents: の後、次の (id:またはTitle:) が来るまでを取得するため、DOTALLを使う
    contents_match = re.search(r'^Contents:\s*(.*?)(?=^id:\s*\d+|^Title:|\Z)', content, flags=re.MULTILINE|re.DOTALL)
    if contents_match:
        # 前後の空白文字をtrim
        contents = contents_match.group(1).strip()
    else:
        contents = None
    return doc_id, title, contents

## データのベクトル化
def text2Vec(data, embedding):
    vec_data = embedding.embed_query(data)
    return vec_data

### CSVファイルが対象の場合
def csv_indexer(file_path, target_file, embeddings):
    loader = CSVLoader(file_path=f"{file_path}/db/csv_index/add_data/{target_file}", encoding='utf-8')
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(docs)
    # documentsリストからDataFrameへ
    data = []
    vec_data = []
    for doc in documents:
        doc_id, title, contents = parse_page_content(doc.page_content)

        row = {
            #'id': doc_id,
            'Title': title,
            'Contents': contents
        }
        # メタデータも展開したい場合
        # for k, v in doc.metadata.items():
        #     row[k] = v
        data.append(row)
    df = pd.DataFrame(data)
    for i in range(len(df)):
        target_text = df.loc[i, "Contents"].replace("\u3000", "")
        target_vec = text2Vec(target_text, embedding)
        vec_data.append(target_vec)
    df["Vector"] = vec_data
    return df

## フォルダ異動
def move_file(file_path, file_name):
    shutil.move(f"{file_path}/db/csv_index/add_data/{file_name}", f'{file_path}/db/csv_index/stored_data')
    print("- 対象ファイルのベクトル化が終了しました。")

## Indexの操作（書き込み）
def check_index(file_path, file_name, df):
    csv_file = f"{file_path}/db/csv_index/vector_index.csv"
    if os.path.exists(csv_file) == True:
        print("-- 既存のIndexに追加します。")
        df.to_csv(csv_file, mode='a', index=False, header=False, encoding='utf-8')
    else:
        df.to_csv(csv_file, index=False, encoding='utf-8')
        print("-- 新規にIndexを作成します。")
    move_file(file_path, file_name)

## 追加されたファイルがIndex化済みか確認
def check_target_before_indexing(files, file_path):
    files = os.listdir(f"{file_path}/db/csv_index/add_data")
    file_list = []
    for i in range(len(files)):
        file_path = os.path.join(f"{file_path}/db/csv_index/stored_data", files[i])
        if os.path.exists(file_path):
            print(f"{file_path} が存在します。")
        else:
            print(f"{files[i]} は指定されたディレクトリに存在しません。")
            file_list.append(files[i])
    return file_list

def main():
    files = os.listdir(f"{FILE_PATH}/db/csv_index/add_data")
    file_check_list = check_target_before_indexing(files, FILE_PATH)
    df_stac = pd.DataFrame()

    if file_check_list != []:
        print("追加されたファイルを確認しました。")
        for i in range(len(file_check_list)):
            target_file = file_check_list[i]
            print(f"{target_file}の処理を開始します。")
            if '.csv' in target_file:
                print("- csvファイルを認識しました。")
                df = csv_indexer(FILE_PATH, target_file, embedding)
                # DataFrameを結合
                df_stac = pd.concat([df_stac, df], ignore_index=True)
            else:
                print("対象外のファイルです。")
            check_index(FILE_PATH, target_file, df_stac)
        print("faissDBを構築開始いたします。")
        # ベクトルデータをロード
        vectors, meta_data = load_csv_and_prepare_vectors(f"{FILE_PATH}/db/csv_index/vector_index.csv")
        # FAISSインデックスを作成し、メタデータを保存
        faiss_index_path = f"{FILE_PATH}/db/csv_index/faiss_dump/faiss_index"
        metadata_path = f"{FILE_PATH}/db/csv_index/faiss_dump/faiss_metadata.txt"
        build_faiss_index_with_metadata(vectors, meta_data, faiss_index_path, metadata_path)
    else:
        print("追加されたファイルは無いようです。")
  
