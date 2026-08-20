from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
import os

# Load environment variables from .env file
load_dotenv()

llm = AzureChatOpenAI(
    # The deployment name of the LLM model to use, set in the .env file
    azure_deployment=os.environ["LLM_MODELS_TURBO"],

    # Version of the OpenAI API to use, set in the .env file
    api_version=os.environ["OPENAI_API_VERSION"],
)

try:
    # This is a simple test to check if the Azure OpenAI API version is supported and working correctly
    response = llm.invoke("Just mention the name of the team: Which is the best soccer team of Peru?")

    print("✅ Azure accept the API version:", os.environ["OPENAI_API_VERSION"])
    print("Answer:", response.content)

except Exception as error:
    # If the API version is not supported, Azure OpenAI will return an error
    print("❌ The API version is not supported by Azure OpenAI. Please check the .env file and ensure the OPENAI_API_VERSION is set correctly.")
    print(error)