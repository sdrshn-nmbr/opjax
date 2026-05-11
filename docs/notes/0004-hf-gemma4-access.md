# Hugging Face Gemma 4 Access Gate

## Current Gate

The first Hugging Face probe ran inside Modal using `opjax-secrets`. Modal wiring worked, but Hugging Face returned `401 Invalid user token` for `HF_TOKEN`.

Local token metadata showed:

- `HF_TOKEN` exists in `.env`
- length is 44
- does not start with `hf_`
- contains whitespace

That means this is not a valid Hugging Face access token shape for the HF Hub API.

## Fix

Update `.env` so the token line is exactly:

```bash
HF_TOKEN=hf_...
```

No quotes, no spaces, no inline comment on the same line.

Then refresh Modal's secret:

```bash
uv run modal secret create -e main opjax-secrets --from-dotenv .env --force
```

Then rerun discovery:

```bash
uv run modal run -e main -m opjax.remote.modal_app::hf_gemma4_discovery_cli
```

## Why This Is The Right First Gate

We do not want to debug Gemma, MaxText, safetensors, or CUDA until the HF auth layer is known-good. The correct sequence is:

- token validity
- model ID discovery
- manifest inspection without blobs
- small config/tokenizer download
- only then weight download
