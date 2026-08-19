import os
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from datetime import datetime

FILE_PATH = os.environ["FILE_PATH"].strip()

today = datetime.now().strftime("%Y年%m月%d日")

embedding = AzureOpenAIEmbeddings(
    openai_api_version="2024-08-01-preview",
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
    azure_deployment=os.environ["EM_MODELS"],
    tiktoken_model_name="text-embedding-3-large",
    check_embedding_ctx_length=False,
)

llm = AzureChatOpenAI(
        openai_api_version="2024-08-01-preview",
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
        deployment_name=os.environ["LLM_MODELS_TURBO"],
        temperature=0
    )

# 2. Cargar el índice FAISS que generaste
vector_store = FAISS.load_local(
    f"{FILE_PATH}/db", embedding,
    allow_dangerous_deserialization=True   # necesario para cargar el .pkl
)

# Cargar el prompt UNA sola vez, fuera del while (no en cada pregunta)
with open(f"{FILE_PATH}/db/prompt_template.txt", "r", encoding="utf-8") as f:
    prompt_template = f.read()

## start

history = ""

# Cuenta blogs únicos a partir de los metadatos del índice FAISS
# (usamos "title" porque cada blog tiene un título único, aunque se divida en varios chunks)
titles_unique = set()
for doc_id, doc in vector_store.docstore._dict.items():
    titulo = doc.metadata.get("title")
    if titulo:
        titles_unique.add(titulo)

current_blogs = len(titles_unique)
print(f"📊 Blogs únicos indexados: {current_blogs}")


print("Starting conversation... (write 'exit' to finish)")

while True:
    question = input("\n質問を入力してください： ")

    if question.lower() in ["exit", "quit", "終了"]:
        print("さようなら！何かあったらまた来てね♪")
        break
    docs = vector_store.similarity_search(question, k=10)   # top 3 chunks

    print("📚 見つけたchunk:")
    context = "" # ← inicializa ANTES del for

    for d in docs:
        print("-", d.page_content[:100], "...") # imprime (recortado)
        date = d.metadata.get("date", "不明") # ← DENTRO del for
        title = d.metadata.get("title", "不明")
        context += f"【投稿日: {date} / タイトル: {title}】\n{d.page_content}\n\n" # acumula los 5 ✅

    prompt = prompt_template.format(
    today=today,
    current_blogs= current_blogs,
    context=context,
    history=history,
    question=question
)

    resp = llm.invoke(prompt)
    print("🤖 社長の回答:", resp.content)

    history += f"質問：{question}\n社長: {resp.content}\n\n"

