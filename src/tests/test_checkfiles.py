import pandas as pd, re
import os
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings

embedding = AzureOpenAIEmbeddings(
        openai_api_version="2024-08-01-preview",
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_deployment=os.environ["EM_MODELS"],
        tiktoken_model_name="text-embedding-3-large",
        check_embedding_ctx_length=False,
    )
vs = FAISS.load_local(f"{os.environ['FILE_PATH']}/data/faiss", embedding,
                          allow_dangerous_deserialization=True)

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

for title, date_str in blogs.items():
    date = parse_date(date_str)
    if date and (last_blog_date is None or date > last_blog_date):
        last_blog, last_blog_date = title, date

current_blogs = len(blogs)

blog_data = pd.read_csv(
    "src/data/knowledge_data.csv",   # adjust the name/path if different
    sep=";", 
    header=0,
    names=["title", "author", "date", "content"],
    engine="python", 
    encoding="utf-8"
)

print("読んだ (ブログ) 行:", blog_data.shape[0])   # ← nº of rows read (means blogs read)

documents = []
for _, row in blog_data.iterrows():
    page = f"author: {row['author']} date: {row['date']} title: {row['title']} content: {row['content']}"
    documents.append(Document(page_content=page))

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = splitter.split_documents(documents)

# Lnength of each blog content
large = blog_data["content"].str.len()

# How many blogs have more than 1000 characters
more_than_1000 = (large > 1000).sum()
total = len(large)

print(f"\n📏 Blogs with MORE than 1000 characters: {more_than_1000} of {total} ({more_than_1000/total*100:.1f}%)")
print(f"📏 Blogs that fit in 1 chunk (≤1000): {total - more_than_1000} ({(total-more_than_1000)/total*100:.1f}%)")

print("Generated chunks:", len(chunks))   # ← the number we're looking for
print(blog_data["content"].str.len().describe())

print(current_blogs, "unique blogs in the database.")
print("Last blog:", last_blog, "with date:", last_blog_date.strftime("%d/%m/%Y %H:%M") if last_blog_date else "N/A")

