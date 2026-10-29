import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter

load_dotenv(Path(__file__).with_name(".env"))

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key or api_key == "paste_your_key_here":
    raise ValueError("Add your OpenRouter API key to .env.")

model = ChatOpenRouter(
    model=os.getenv("OPENROUTER_MODEL", "openrouter/free"),
    temperature=0,
    max_retries=0,
)

response = model.invoke(
    "Reply with exactly: LLM connection successful"
)

print(response.content)