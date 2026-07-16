#!/usr/bin/env bash
# Cloud / VM bootstrap for opjax Model Factory (Wave A) or full opjax.
#
# Usage:
#   ./scripts/setup_cloud.sh              # factory venv (tinker/hf/prime) — default
#   ./scripts/setup_cloud.sh --full       # submodule + uv sync (JAX/tunix/gemma)
#   ./scripts/setup_cloud.sh --check      # secrets presence only
#
# Secrets: copy .env.example → .env, or inject via Cursor Cloud Environment.
# This script never prints secret values.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="factory"
CHECK_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --full) MODE="full" ;;
    --factory) MODE="factory" ;;
    --check) CHECK_ONLY=1 ;;
    -h|--help)
      sed -n '2,12p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown arg: $arg" >&2
      exit 2
      ;;
  esac
done

# Load .env if present (without exporting into process list dumps via set -x)
if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
  echo "Loaded .env"
else
  echo "No .env found (ok if Cursor Environment injects secrets)."
fi

# Normalize Cursor Cloud / alternate secret names → project conventions
if [[ -z "${HF_TOKEN:-}" && -n "${HUGGINGFACE_TOKEN:-}" ]]; then
  export HF_TOKEN="$HUGGINGFACE_TOKEN"
fi
if [[ -z "${HF_TOKEN:-}" && -n "${HF_API_KEY:-}" ]]; then
  export HF_TOKEN="$HF_API_KEY"
fi
if [[ -z "${PRIME_API_KEY:-}" && -n "${PRIMEINTELLECT_API_KEY:-}" ]]; then
  export PRIME_API_KEY="$PRIMEINTELLECT_API_KEY"
fi
if [[ -z "${PRIME_API_KEY:-}" && -n "${PRIME_KEY:-}" ]]; then
  export PRIME_API_KEY="$PRIME_KEY"
fi

check_secrets() {
  echo ""
  echo "=== Secret presence (values hidden) ==="
  local required=(HF_TOKEN TINKER_API_KEY)
  local optional=(PRIME_API_KEY ANTHROPIC_API_KEY OPENAI_API_KEY WANDB_API_KEY TOGETHER_API_KEY MODAL_TOKEN_ID MODAL_TOKEN_SECRET)
  local missing=0
  for k in "${required[@]}"; do
    if [[ -n "${!k:-}" ]]; then
      echo "  OK   $k"
    else
      echo "  MISS $k  (required for Wave A trainer/Hub)"
      missing=1
    fi
  done
  for k in "${optional[@]}"; do
    if [[ -n "${!k:-}" ]]; then
      echo "  OK   $k"
    else
      echo "  --   $k  (optional)"
    fi
  done
  if command -v prime >/dev/null 2>&1; then
    # `prime config view` always exits 0; inspect for a real key / user id.
    local pcfg
    pcfg="$(prime --plain config view 2>/dev/null || prime config view 2>/dev/null || true)"
    if [[ -n "${PRIME_API_KEY:-}" ]]; then
      echo "  OK   prime (PRIME_API_KEY env)"
    elif echo "$pcfg" | grep -qiE 'API Key[[:space:]]+Not set|User ID[[:space:]]+Not set'; then
      echo "  --   prime CLI installed but not logged in (run: prime login)"
    elif echo "$pcfg" | grep -qiE 'API Key|User ID'; then
      echo "  OK   prime CLI session"
    else
      echo "  --   prime CLI installed; login status unknown (run: prime login)"
    fi
  else
    echo "  --   prime CLI not installed yet"
  fi
  return "$missing"
}

if [[ "$CHECK_ONLY" -eq 1 ]]; then
  check_secrets || true
  exit 0
fi

ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    echo "uv: $(uv --version)"
    return
  fi
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  hash -r
  uv --version
}

ensure_uv

if [[ "$MODE" == "full" ]]; then
  echo "=== Full opjax sync (submodule + uv sync) ==="
  git submodule update --init --recursive references/tunix
  uv sync --group dev
  echo "Installing Wave A extras into project venv..."
  uv pip install tinker tinker-cookbook
else
  echo "=== Factory venv (Wave A: tinker / HF / light deps) ==="
  # Isolated venv so we do not need tunix/gemma for cloud auth smoke.
  if [[ -d .venv && -x .venv/bin/python ]]; then
    echo "Reusing existing .venv"
  else
    uv venv .venv
  fi
  # shellcheck disable=SC1091
  source .venv/bin/activate
  uv pip install \
    "huggingface-hub>=0.36" \
    "tinker>=0.23" \
    "tinker-cookbook>=0.5" \
    "anthropic>=0.40" \
    "python-dotenv>=1.0" \
    "hf_transfer>=0.1" \
    "pyyaml>=6.0"
  # Inkling renderer + tokenizer bridge (tml_renderers)
  uv pip install 'tinker-cookbook[inkling]'
fi

echo "Installing prime CLI (uv tool)..."
uv tool install -U prime

echo ""
echo "=== Smoke (no private data) ==="
export PATH="$HOME/.local/bin:$PATH"
# shellcheck disable=SC1091
[[ -f "$ROOT/.venv/bin/activate" ]] && source "$ROOT/.venv/bin/activate"
# Factory package lives in src/ without requiring full `uv sync` (JAX/tunix).
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# Full mode also needs inkling renderers for Model Factory SFT
if [[ "$MODE" == "full" ]]; then
  uv pip install 'tinker-cookbook[inkling]' || true
fi

python - <<'PY'
import importlib
for m in ("tinker", "huggingface_hub", "anthropic", "tml_renderers"):
    try:
        mod = importlib.import_module(m)
        ver = getattr(mod, "__version__", "?")
        print(f"  OK import {m} ({ver})")
    except Exception as e:
        print(f"  FAIL import {m}: {e}")
PY

if command -v tinker >/dev/null 2>&1; then
  echo "  OK tinker CLI: $(command -v tinker)"
else
  # cookbook/sdk may expose module CLI differently
  python -c "import tinker; print('  OK tinker module', getattr(tinker, '__version__', ''))"
fi

echo "  prime: $(command -v prime || echo missing)"
if command -v hf >/dev/null 2>&1; then
  echo "  OK hf CLI: $(command -v hf)"
else
  echo "  -- hf CLI not on PATH (huggingface_hub login fallback OK)"
fi

check_secrets || true

echo ""
echo "=== Next ==="
echo "  1. Put secrets in .env or Cursor Cloud Environment (HF_TOKEN, TINKER_API_KEY)."
echo "  2. source .venv/bin/activate   # or: set -a && source .env && set +a"
echo "  3. hf auth login --token \"\$HF_TOKEN\"   # once per VM (or ./scripts/env_update.sh)"
echo "  4. prime login                 # browser or API key"
echo "  5. Optional: tinker run list / ServiceClient smoke — no private uploads yet"
echo "Done ($MODE)."
