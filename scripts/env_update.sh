#!/usr/bin/env bash
# Paste-friendly Cursor Cloud "Update script".
# Idempotent: safe to re-run on every agent start from snapshot.
#
# Expected Cloud secrets (any of the aliases work):
#   HUGGINGFACE_TOKEN or HF_TOKEN
#   TINKER_API_KEY
#   PRIMEINTELLECT_API_KEY or PRIME_API_KEY  (optional but recommended)
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Optional local .env — do not fail if missing (Cloud injects secrets as env)
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

# Ensure packages exist (no-op if already installed)
./scripts/setup_cloud.sh --factory

# shellcheck disable=SC1091
source .venv/bin/activate

# Aliases (also applied inside setup_cloud.sh)
export HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_TOKEN:-${HF_API_KEY:-}}}"
export PRIME_API_KEY="${PRIME_API_KEY:-${PRIMEINTELLECT_API_KEY:-}}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN / HUGGINGFACE_TOKEN not set" >&2
  exit 1
fi
if [[ -z "${TINKER_API_KEY:-}" ]]; then
  echo "ERROR: TINKER_API_KEY not set" >&2
  exit 1
fi

huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential

if [[ -n "${PRIME_API_KEY:-}" ]]; then
  prime --plain config set-api-key "$PRIME_API_KEY"
else
  echo "WARN: no PRIME_API_KEY / PRIMEINTELLECT_API_KEY — skip prime auth (use prime login interactively)"
fi

./scripts/setup_cloud.sh --check
echo "env_update.sh OK"
