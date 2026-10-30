"""Run from the project root: python -m test_files.check_llm"""

from llm_config import create_model


if __name__ == "__main__":
    model = create_model(cache=False)
    response = model.invoke("Reply with exactly: LLM connection successful")
    print(response.content)
