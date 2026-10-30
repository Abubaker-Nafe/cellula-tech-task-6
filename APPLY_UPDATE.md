# Apply the required Task 6 updates

This is an overlay for your existing Task 6 project. It is not a complete clone.
The `updated_files` folder contains the files to add or replace, including the
updated Generator/Evaluator, shared helpers, ingestion changes, Cache panel,
README, and tests. It includes the previous required fixes as well, plus the
configurable Redis connection for an already deployed Streamlit application.

## Already deployed on Streamlit Community Cloud?

Follow `updated_files/STREAMLIT_DEPLOYMENT.md`. The deployment connection fix needs
new `redis_config.py`, updated `app.py`, `generator.py`, and `evaluator.py`, plus
`packages.txt` for cloud LibreOffice installation. The diagnostic, tests, and docs
are updated too. Keep your existing OpenRouter secrets and add a hosted Redis
`REDIS_URL` to Streamlit Secrets. Do not copy a Windows LibreOffice path to the
cloud, and do not expose your local Docker Redis to the internet. No new Python
dependencies are needed for this connection change.

If the earlier update is already installed, the Cache panel needs the updated
`app.py`, `workflow.py`, `llm_cache.py`, and the new `cache_monitor.py`. Copying all
of `updated_files` also updates the tests and documentation. No new Python
dependencies are needed for the Cache panel itself.

## 1. Stop and back up

Stop Streamlit with **Ctrl+C**. Make a backup copy of your existing project files
before replacing them. In particular, retain your real `.env`, documents, audio,
Docker Compose file, and any extra local changes.

The unchanged baseline came from public repository commit
`e0905669df39496f381f5e95edb586009dcc3900` (October 29, 2026), combined with the
Evaluator/cache corrections discussed in this conversation. If you have made
additional unrelated changes, compare them before replacing the listed files.

## 2. Copy the update

Copy the **contents** of `updated_files` into:

```text
C:\Users\nafea\Desktop\Cellula Tech\Week-6\Task-6
```

Merge the `test_files` folder and replace matching files when prompted. Do not
place the entire `updated_files` folder inside the project. Your project root
should directly contain `llm_config.py`, `evaluation_schema.py`, `embedding_config.py`,
and `ppt_ingestion.py` beside `app.py`.

This archive contains an `.env.example`, not a real `.env`. Retain your actual API
key and this existing model line:

```dotenv
OPENROUTER_MODEL=dots-studio/dots-3-note-preview:free
```

## 3. Install the legacy PPT converter

Install the Windows x86-64 version of LibreOffice from:
https://www.libreoffice.org/download/download-libreoffice/

LibreOffice is required for direct binary `.ppt` support. `.pptx` uses python-pptx
and does not need the converter. Standard Windows installations are detected
automatically. If detection fails, add this line to your `.env` using the actual path:

```dotenv
LIBREOFFICE_PATH=C:/Program Files/LibreOffice/program/soffice.com
```

Restart the VS Code terminal after installing LibreOffice, then activate `.venv`.

## 4. Install/check the Python dependencies

Run from your project root in **Command Prompt**, with `.venv` active:

```cmd
python -m pip install -r requirements.txt
python -m pip check
python -m unittest discover -s tests -v
python -m test_files.check_memory
python ingestion.py test_files\fixtures\legacy_demo.ppt
```

Expected: `pip check` reports no broken requirements; the 29 regression tests end
with `OK`; memories remain separate; the PPT fixture produces two slide chunks
mentioning a four-iteration limit and Redis caching. Unit tests use controlled
provider/Redis doubles, and do not consume your real API quota or touch Redis data.
The PPT command uses LibreOffice and the embedding tokenizer; its first use can
download tokenizer files.

## 5. Start and check the application

Keep Docker Desktop running, then:

```cmd
docker compose up -d
python -m streamlit run app.py --server.address 127.0.0.1
```

Process your sources again; token-based chunking can change chunk counts and
retrieval results. Upload `test_files\fixtures\legacy_demo.ppt` to verify direct
PPT support, or reuse your PDFs and URL.

The updated Evaluator uses `task6:evaluator:v3`. Its first request can miss even
when an old evaluation was cached. Entries in earlier namespaces expire naturally;
no Redis clearing is needed. To see hits, use the same inputs with empty histories:
clear conversation, ask, wait, clear again, then paste the identical question.

Open **Cache** below an answer to see Generator/Evaluator HIT or MISS for each
round. Existing answers retain their original lookup results when the UI reruns.
Repeated questions inside the same conversation can still miss: the cache also
matches conversation history, retrieved evidence, and model settings. Clearing
history recreates the original model request; it does not clear Redis.

The offline test suite and real tokenizer/conversion checks passed in a Linux
Python 3.12.14 environment using the pinned LangChain/OpenRouter versions. Live
OpenRouter inference, your Windows installation, and a complete Streamlit UI run
were not performed here. The smoke checks above verify those local prerequisites.

## Scope

The optional features from the review are not part of this package. This update
adds the requested Cache display and configurable Redis deployment endpoint while
retaining full-request cache matching. The local Docker URL remains the default.

The four-round limit, early exit, separate memories, normal feedback flow, and
invalid-assessment fallback are retained. Detailed architecture and limitations
are documented in the included README.
