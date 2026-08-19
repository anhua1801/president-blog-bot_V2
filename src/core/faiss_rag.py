import re
import os
from flask import Response, stream_with_context
from langchain.vectorstores.faiss import FAISS
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()
LLM_MODELS = os.environ["LLM_MODELS"]
FILE_PATH = os.environ["FILE_PATH"]
MODE = os.environ["MODE"]

class RAGHandler:
    def __init__(self, embedding, client, character_prompt, system_prompt):
        if MODE == "csv_index":
            self.retriever =None
        else:
            vector_store = FAISS.load_local(f"{FILE_PATH}/db/faiss", embeddings=embedding, allow_dangerous_deserialization=True)
            self.retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        self.client = client
        self.character_prompt = character_prompt
        self.system_prompt_template = system_prompt
        self.response_data = []

    def fetch_relevant_docs(self, user_message):
        retrieved_docs = self.retriever.invoke(user_message)
        contents = [
            re.search(r"Contents:\s*(.*)", doc.page_content).group(1)
            for doc in retrieved_docs if re.search(r"Contents:\s*(.*)", doc.page_content)
        ]
        return contents

    def update_system_prompt(self, contents):
        contents_str = "[\n" + ",\n".join(contents) + "\n]"
        return self.system_prompt_template.replace("nearly_contens", contents_str)

    def generate_stream(self, user_message, updated_system_prompt):
        stream = self.client.chat.completions.create(
            model=LLM_MODELS,
            messages=[
                {"role": "system", "content": self.character_prompt},
                {"role": "system", "content": updated_system_prompt},
                *self.response_data,
                {"role": "user", "content": user_message}
            ],
            stream=True
        )
        answers = ""
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content is not None:
                answers += chunk.choices[0].delta.content.encode('utf-8').decode('utf-8')
                yield chunk.choices[0].delta.content.encode('utf-8')
        self.update_response_data(user_message, updated_system_prompt, answers)

    def update_response_data(self, user_message, updated_system_prompt, answers):
        self.response_data.append({"role": "user", "content": user_message})
        self.response_data.append({"role": "system", "content": updated_system_prompt})
        self.response_data.append({"role": "assistant", "content": answers})
        if len(self.response_data) > 15:
            del self.response_data[:3]  # 古いデータを削除
