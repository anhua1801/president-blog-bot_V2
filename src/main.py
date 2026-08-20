from flask import Flask, request, Response, render_template, stream_with_context
from core.faiss_rag import RAGHandler, load_bot
from core.csv_db_rag import search_Index 
from utils.check_file import check_file_exists
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from openai import AzureOpenAI
import os
import datetime
from dotenv import load_dotenv

load_dotenv()

# Initialize environment variables
FILE_PATH = os.environ["FILE_PATH"]
AZURE_OPENAI_API_KEY = os.environ["AZURE_OPENAI_API_KEY"]
AZURE_OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
LLM_MODELS = os.environ["LLM_MODELS"]
EM_MODELS = os.environ["EM_MODELS"]
MODE = os.environ["MODE"]

# Initialize models and vector store
embedding = AzureOpenAIEmbeddings(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        azure_deployment=os.environ["EM_MODELS"],
        tiktoken_model_name="text-embedding-3-large",
        check_embedding_ctx_length=False
        )

llm = AzureChatOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        azure_deployment=os.environ["LLM_MODELS_TURBO"],
        temperature=0)

# Load prompts
## botのキャラクター性に関するもの
character_prompt_file_path = 'prompts/character_prompts.txt'
if check_file_exists(character_prompt_file_path):
    print("character_prompts ファイルが存在します")
    with open(f'{FILE_PATH}/prompts/character_prompts.txt', 'r', encoding='utf-8') as f:
        character_prompt = f.read()
else:
    print("character_prompts ファイルが存在しません")
    with open(f'{FILE_PATH}/prompts/character_prompts_Sample.txt', 'r', encoding='utf-8') as f:
        character_prompt = f.read()

## systemの設定
sys_prompt_file_path = 'prompts/system_prompts.txt'
if check_file_exists(sys_prompt_file_path):
    print("system_prompts ファイルが存在します")
    with open(f'{FILE_PATH}/prompts/system_prompts.txt', 'r', encoding='utf-8') as f:
        system_prompt = f.read()
else:
    print("system_prompts ファイルが存在しません")
    with open(f'{FILE_PATH}/prompts/system_prompts_Sample.txt', 'r', encoding='utf-8') as f:
        system_prompt = f.read()
system_prompt = system_prompt.replace("today_date",(datetime.date.today()).strftime("%Y/%m/%d"))

# Initialize RAGHandler
bot = load_bot(embedding, llm, character_prompt, system_prompt)

# Flask app setup
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json['message']
    if MODE == "csv_index":
        contents = search_Index(user_message, embedding)
    else:
        contents = bot.fetch_relevant_docs(user_message)
    updated_system_prompt = bot.update_system_prompt(contents)

    def generate():
        yield from bot.generate_stream(user_message, updated_system_prompt)

    return Response(stream_with_context(generate()), content_type='text/plain')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)