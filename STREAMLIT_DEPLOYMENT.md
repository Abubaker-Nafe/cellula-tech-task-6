# Connect the deployed Streamlit app to Redis

The deployed application runs on another computer. Its `127.0.0.1` address does
not point to your Windows PC or your Docker container. Keep that container for
local development; the deployed app uses a hosted Redis endpoint through
`REDIS_URL`. Existing local cache entries do not need to be migrated.

## 1. Create a hosted Redis database

One free option for this demonstration is Upstash Redis:
https://console.upstash.com/

1. Sign in and create a **Redis** database, for example `task6-cache`.
2. Choose a region near your Streamlit app's server and the **Free** plan.
3. Use the normal Redis/TCP connection details, with **TLS enabled**. Copy its
   connection URL if offered. Otherwise use the host, port, and database password
   from the Redis CLI connection shown on its Details/Connect screen.

The expected URL format is:

```text
rediss://default:YOUR_DATABASE_PASSWORD@YOUR_DATABASE_HOST:YOUR_DATABASE_PORT/0
```

The extra `s` in `rediss://` enables TLS. Use the actual port from the dashboard.
Do not put an `https://` REST endpoint or an `UPSTASH_REDIS_REST_URL` into
`REDIS_URL`: this project uses the existing `redis` Python client over Redis/TCP.
If constructing the URL manually, percent-encode special characters in its
username/password. Prefer copying a correctly encoded URL from the provider.

Our pinned `langchain-redis` cache requires **JSON.SET and JSON.GET**, not only
PING/GET/SET. Upstash documents these JSON commands. Another Redis provider is
also usable if it supports these commands and an appropriate TLS connection.
The project does not require a separate Redis Search index for this exact cache.

As checked on October 30, 2026, Upstash's Free plan lists 256 MB of data and
500,000 commands per month. Stay on the Free plan for this free demonstration;
adding a payment method changes its billing behavior. Check the current plan
limits before enabling paid options.

## 2. Apply and push the code update

Copy the contents of this package's `updated_files` folder into your project root.
This is an overlay and does not contain your complete project. Keep your `.env`
and any unrelated customizations. If prior updates are already applied, the
deployment-specific changes are:

- New `redis_config.py` shared connection settings.
- Updated `generator.py`, `evaluator.py`, and `app.py` using that configuration.
- Updated `test_files/check_redis.py` checking both connectivity and JSON support.
- New `packages.txt` installing LibreOffice Impress for legacy PPT conversion.
- Updated `.env.example`, regression tests, and deployment documentation.

Run offline checks locally with your existing virtual environment activated:

```cmd
python -m unittest discover -s tests -v
```

Push the changed code to the GitHub branch used by your Streamlit deployment.
Include `redis_config.py` and root `packages.txt`. Keep real credentials in
`.env`/Streamlit Secrets rather than committing them. No additional Python
dependency is needed for the Redis connection change.

## 3. Set Streamlit Secrets

In your deployed app's **Settings -> Secrets**, retain your existing OpenRouter
settings and add the top-level `REDIS_URL`:

```toml
OPENROUTER_API_KEY = "your_existing_key"
OPENROUTER_MODEL = "dots-studio/dots-3-note-preview:free"
REDIS_URL = "rediss://default:YOUR_DATABASE_PASSWORD@YOUR_DATABASE_HOST:YOUR_DATABASE_PORT/0"
```

Replace placeholders with your real provider values. Do not paste a password or
the complete URL into chat or publish it in a screenshot. Streamlit exports
top-level secrets as environment variables, which this code reads.

Do not add your Windows `LIBREOFFICE_PATH` to the cloud secrets. Streamlit
installs LibreOffice from root `packages.txt`; the converter locates it through
the Linux PATH. Your Windows `.env` can retain its local LibreOffice setting.

Save the secrets and reboot the app after the code/dependency update is built.
The app must be able to connect to the hosted Redis endpoint. If you enable a
provider IP allowlist, it must permit the deployed server's outbound traffic;
allowing only your PC's IP will not permit the cloud app to connect.

## 4. Verify the deployed app

1. Confirm the Redis connection error is gone.
2. Process your sources and ask an explicit question. A fresh database starts
   with empty caches, so the first lookup normally misses.
3. Clear the conversation and paste the identical question with the sources
   unchanged. The answer's **Cache** expander should show HIT for both agents
   while the entries remain valid. Repeating inside an existing conversation
   can still miss because history is part of the model request.
4. For direct PPT ingestion, upload `test_files/fixtures/legacy_demo.ppt` and ask
   about its maximum feedback-loop iteration count. It should answer four.

To check the same hosted Redis from Windows, temporarily set its URL in your
local `.env`, then run:

```cmd
python -m test_files.check_redis
```

Expected output:

```text
Connected: True
Redis JSON support: True
Stored value: Redis is working
```

The check creates a uniquely named probe, sets a short TTL, and deletes that
probe. It does not clear application caches. Restore the local URL afterward
if you want your PC to keep using Docker:

```dotenv
REDIS_URL=redis://127.0.0.1:6379/0
```

If `REDIS_URL` is absent, that local Docker URL is still the default. Cloud
secrets take priority over a dotenv value because dotenv does not override
existing environment variables.

## References

- Upstash setup and TLS connection:
  https://upstash.com/docs/redis/overall/getstarted
- Upstash JSON commands:
  https://upstash.com/docs/redis/sdks/py/commands/json/set
  https://upstash.com/docs/redis/sdks/py/commands/json/get
- Free-plan limits and billing:
  https://upstash.com/pricing/redis
- Redis URL syntax:
  https://redis.io/docs/latest/develop/clients/connection-urls/
- Streamlit Secrets:
  https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management
- Streamlit system packages:
  https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies
