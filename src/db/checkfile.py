import pandas as pd, re
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

blog_data = pd.read_csv(
    "knowledge_data.csv",   # ajusta el nombre/ruta si es distinto
    sep=";", 
    header=0,
    names=["author", "date", "title", "content"],
    engine="python", 
    encoding="utf-8"
)

print("読んだ (ブログ) 行:", blog_data.shape[0])   # ← nº de blogs

documents = []
for _, row in blog_data.iterrows():
    page = f"author: {row['author']} date: {row['date']} title: {row['title']} content: {row['content']}"
    documents.append(Document(page_content=page))

splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100)
chunks = splitter.split_documents(documents)

# Largo de cada blog (en caracteres)
large = blog_data["content"].str.len()

# ¿Cuántos pasan de 512?
more_than_512 = (large > 512).sum()
total = len(large)

print(f"\n📏 Blogs con MÁS de 512 caracteres: {more_than_512} de {total} ({more_than_512/total*100:.1f}%)")
print(f"📏 Blogs que caben en 1 chunk (≤512): {total - more_than_512} ({(total-more_than_512)/total*100:.1f}%)")

print("Chunks generados:", len(chunks))   # ← el número que buscamos
print(blog_data["content"].str.len().describe())

