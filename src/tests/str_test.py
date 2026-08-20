import streamlit as st
import os
from datetime import datetime
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import pandas as pd
from dotenv import load_dotenv

# --- Carga única (cacheada para no recargar en cada mensaje) ---

load_dotenv() #Carga el env de un archivo .env en el directorio raíz del proyecto

FILE_PATH = os.environ["FILE_PATH"].strip()

today = datetime.now().strftime("%Y年%m月%d日")

with open(f"{FILE_PATH}/prompts/prompt_template.txt", "r", encoding="utf-8") as f:
    prompt_template = f.read()

@st.cache_resource
def cargar_bot():
    embedding = AzureOpenAIEmbeddings(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        azure_deployment=os.environ["EM_MODELS"],
        tiktoken_model_name="text-embedding-3-large",
        check_embedding_ctx_length=False
    )
    vs = FAISS.load_local(f"{os.environ['FILE_PATH']}/data/faiss", embedding, allow_dangerous_deserialization=True)
    llm = AzureChatOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        azure_deployment=os.environ["LLM_MODELS_TURBO"],
        temperature=0)
    return vs, llm

vs, llm = cargar_bot()

st.title("🎩 社長Bot - 倉川社長とお話しましょう")

# Historial en la sesión
if "messages" not in st.session_state:
    st.session_state.messages = []

# Muestra el historial
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Last and current blogs
blogs = {} # title -> date_string, to avoid duplicates due to chunks
for doc_id, doc in vs.docstore._dict.items():
    title = doc.metadata.get("title")
    date = doc.metadata.get("date")
    if title and date:
        blogs[title] = date  # Save the latest date for each title, overwriting duplicates

def parse_date(date_str):
    try:
        return pd.to_datetime(date_str, format="%d/%m/%Y %H:%M")
    except ValueError:
        return None

last_blog, last_blog_date = None, None

current_blogs = len(blogs)

for title, date_str in blogs.items():
    date = parse_date(date_str)
    if date and (last_blog_date is None or date > last_blog_date):
        last_blog, last_blog_date = title, date

# Guardar el historial para darselo al bot

history = "\n".join(
    [f"{m['role']}: {m['content']}" for m in st.session_state.messages]
)
# Input del usuario
if question := st.chat_input("質問を入力してください..."):
    st.chat_message("user").write(question)
    st.session_state.messages.append({"role": "user", "content": question})

    # Buscar + armar contexto
    docs = vs.similarity_search(question, k=10)
    context = ""
    for d in docs:
        context += f"【{d.metadata.get('date','?')} / {d.metadata.get('title','?')}】\n{d.page_content}\n\n"

    today = datetime.now().strftime("%Y年%m月%d日")
    prompt = prompt_template.format(
        today=today,
        current_blogs= current_blogs,
        last_blog=last_blog,
        last_blog_date=last_blog_date.strftime("%d/%m/%Y %H:%M") if last_blog_date else "N/A",
        context=context,
        history=history,
        question=question
    )
    resp = llm.invoke(prompt)
    st.chat_message("assistant").write(resp.content)
    st.session_state.messages.append({"role": "assistant", "content": resp.content})