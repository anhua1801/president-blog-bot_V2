import os
from dotenv import load_dotenv

load_dotenv()

# Environment variables
FILE_PATH = os.environ["FILE_PATH"]

def check_file_exists(file_path):
    return os.path.isfile(f"{FILE_PATH}/{file_path}")