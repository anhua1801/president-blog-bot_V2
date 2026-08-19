import streamlit as st
import os
from datetime import datetime
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# --- Carga única (cacheada para no recargar en cada mensaje) ---

FILE_PATH = os.environ["FILE_PATH"].strip()

today = datetime.now().strftime("%Y年%m月%d日")

with open(f"{FILE_PATH}/db/prompt_template.txt", "r", encoding="utf-8") as f:
    prompt_template = f.read()

@st.cache_resource
def cargar_bot():
    embedding = AzureOpenAIEmbeddings(
        openai_api_version="2024-08-01-preview",
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_deployment=os.environ["EM_MODELS"],
        tiktoken_model_name="text-embedding-3-large",
        check_embedding_ctx_length=False,
    )
    vs = FAISS.load_local(f"{os.environ['FILE_PATH']}/db", embedding,
                          allow_dangerous_deserialization=True)
    llm = AzureChatOpenAI(
        openai_api_version="2024-08-01-preview",
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
        deployment_name=os.environ["LLM_MODELS_TURBO"],
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

# Actuales Blogs

titles_unique = set()
for doc_id, doc in vs.docstore._dict.items():
    titulo = doc.metadata.get("title")
    if titulo:
        titles_unique.add(titulo)

# Guardar el historial para darselo al bot

history = "\n".join(
    [f"{m['role']}: {m['content']}" for m in st.session_state.messages]
)

current_blogs = len(titles_unique)

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
        context=context,
        history=history,
        question=question
    )
    resp = llm.invoke(prompt)
    st.chat_message("assistant").write(resp.content)
    st.session_state.messages.append({"role": "assistant", "content": resp.content})