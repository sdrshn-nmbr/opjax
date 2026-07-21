#!/usr/bin/env bash
# Paste-friendly Cursor Cloud "Update script".
# Idempotent: safe to re-run on every agent start from snapshot.
#
# Expected Cloud secrets (any of the aliases work):
#   HUGGINGFACE_TOKEN or HF_TOKEN
#   TINKER_API_KEY
#   PRIMEINTELLECT_API_KEY or PRIME_API_KEY  (optional but recommended)
#   R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY (axport read)
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Optional local .env — do not fail if missing (Cloud injects secrets as env)
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

# Map Cloud secret names before any tool/auth step
export HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_TOKEN:-${HF_API_KEY:-}}}"
export PRIME_API_KEY="${PRIME_API_KEY:-${PRIMEINTELLECT_API_KEY:-${PRIME_KEY:-}}}"

# Ensure packages exist (no-op if already installed)
./scripts/setup_cloud.sh --factory

# shellcheck disable=SC1091
source .venv/bin/activate

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN / HUGGINGFACE_TOKEN not set" >&2
  exit 1
fi
if [[ -z "${TINKER_API_KEY:-}" ]]; then
  echo "ERROR: TINKER_API_KEY not set" >&2
  exit 1
fi

# huggingface-hub>=1.x ships `hf`; `huggingface-cli` is a deprecated stub that exits 1.
if command -v hf >/dev/null 2>&1; then
  # Prefer env-token auth; disk login is optional persistence for tools that ignore HF_TOKEN.
  # Drop --add-to-git-credential when no credential.helper is configured (common on Cloud VMs).
  if git config --get credential.helper >/dev/null 2>&1; then
    hf auth login --token "$HF_TOKEN" --add-to-git-credential
  else
    hf auth login --token "$HF_TOKEN"
  fi
  hf auth whoami
else
  echo "ERROR: hf CLI missing after setup_cloud.sh (expected via huggingface-hub)" >&2
  exit 1
fi

if [[ -n "${PRIME_API_KEY:-}" ]]; then
  # Non-interactive; Cloud has no browser for `prime login`
  prime --plain config set-api-key "$PRIME_API_KEY" || {
    echo "WARN: prime config set-api-key failed; PRIME_API_KEY remains in env for API clients" >&2
  }
else
  echo "WARN: no PRIME_API_KEY / PRIMEINTELLECT_API_KEY — skip prime auth (use prime login interactively)"
fi

./scripts/setup_cloud.sh --check
echo "env_update.sh OK"
