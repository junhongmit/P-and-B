import os
from os.path import join, dirname
from dotenv import load_dotenv

env_file = os.getenv("ENV_FILE", ".env")
dotenv_path = os.path.join(os.path.dirname(__file__), "..", env_file)
load_dotenv(dotenv_path)

# If API base url is not set, fallback to use vLLM based local server
API_KEY = os.environ.get("API_KEY", "")
API_BASE = os.environ.get("API_BASE")
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")

EMB_API_BASE = os.environ.get("EMB_API_BASE", "http://localhost:7878/v1")
EMB_MODEL_NAME = os.environ.get("EMB_MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct-e")
EMB_CONTEXT_LENGTH = int(os.environ.get("EMB_CONTEXT_LENGTH", "512"))

DATASET_PATH = os.environ.get("DATASET_PATH", "")