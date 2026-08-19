import pandas as pd
import os
import re
from langchain_community.vectorstores import FAISS  # ベクトル検索（近傍探査）
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings # AzureOpenAIEmbeddings 追加
#from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_core.documents import Document

FILE_PATH = os.environ["FILE_PATH"].strip()
file_path = f"{FILE_PATH}/db/knowledge_data.csv"


AZURE_OPENAI_API_KEY = os.environ["AZURE_OPENAI_API_KEY"]
AZURE_OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
LLM_MODELS_TURBO = os.environ["LLM_MODELS_TURBO"]
EM_MODELS = os.environ["EM_MODELS"]

llm = AzureChatOpenAI(openai_api_version="2024-08-01-preview",
                      azure_endpoint=AZURE_OPENAI_ENDPOINT,
                      openai_api_type="azure",
                      openai_api_key=AZURE_OPENAI_API_KEY,
                      deployment_name=LLM_MODELS_TURBO,
                      temperature=0)


embedding = AzureOpenAIEmbeddings(openai_api_version="2024-08-01-preview",
                                  azure_endpoint=AZURE_OPENAI_ENDPOINT,
                                  openai_api_type="azure",
                                  openai_api_key=AZURE_OPENAI_API_KEY,
                                  azure_deployment=EM_MODELS,
                                  tiktoken_model_name="text-embedding-3-large",
                                  check_embedding_ctx_length=False,)

# knowledge chunking
def chunking_data(file_path):
    
    blog_data = pd.read_csv(
        file_path,
        sep=";",
        header=0,
        names=["title", "author", "date", "content"],
        engine="python",
        encoding="utf-8"
    )

    documents = []

    for _, row in blog_data.iterrows():
        author = str(row["author"]) if pd.notnull(row["author"]) else ""
        date = str(row["date"]) if pd.notnull(row["date"]) else ""
        title = str(row["title"]) if pd.notnull(row["title"]) else ""
        content = str(row["content"]) if pd.notnull(row["content"]) else ""

        page_content = f"author: {author} date: {date} title: {title} content: {content}"

        cleaned_content = page_content.replace("\n", "").replace("\u3000", "").replace("   ", "")
        cleaned_content = re.sub(r"^\d+\s", "", cleaned_content)

        documents.append(
            Document(
                page_content=cleaned_content,
                metadata={"author": author, "date": date, "title": title}
            )
        )

    with open(f"{FILE_PATH}/db/knowledge_logs.txt", mode="w", encoding="utf-8") as file:
        file.write(str(documents))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )

    splitted_documents = text_splitter.split_documents(documents)

    return splitted_documents
    
    
splitted_documents = chunking_data(file_path)

# --- Generar embeddings por lotes ---
batch_size = 100
vector_store = None

for i in range(0, len(splitted_documents), batch_size):
    batch = splitted_documents[i:i + batch_size]
    print(f"Procesando {i+len(batch)}/{len(splitted_documents)} chunks...")

    if vector_store is None:
        vector_store = FAISS.from_documents(documents=batch, embedding=embedding)   # 1ra tanda: crea
    else:
        vector_store.add_documents(batch)                                            # resto: agrega

vector_store.save_local(f"{FILE_PATH}/db")   # guarda igual que antes, al final
print("¡Índice guardado! ✅")
