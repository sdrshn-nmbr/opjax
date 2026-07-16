#!/usr/bin/env bash
# Smoke-test Model Factory Wave A toolchain + axport R2 read access.
# Never prints secret values. Exit 0 only if required checks pass.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PATH="$ROOT/.venv/bin:$HOME/.local/bin:$PATH"
# shellcheck disable=SC1091
[[ -f "$ROOT/.venv/bin/activate" ]] && source "$ROOT/.venv/bin/activate"

if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

export HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_TOKEN:-${HF_API_KEY:-}}}"
export PRIME_API_KEY="${PRIME_API_KEY:-${PRIMEINTELLECT_API_KEY:-${PRIME_KEY:-}}}"

FAIL=0
pass() { echo "  PASS  $*"; }
fail() { echo "  FAIL  $*"; FAIL=1; }
info() { echo "  --    $*"; }

echo "=== Toolchain smoke ($(date -u +%Y-%m-%dT%H:%MZ)) ==="

# ── Imports / CLIs ────────────────────────────────────────────────────────
python - <<'PY' || fail "python imports"
import importlib
for m in ("tinker", "huggingface_hub"):
    mod = importlib.import_module(m)
    print(f"  PASS  import {m} ({getattr(mod, '__version__', '?')})")
PY

if command -v tinker >/dev/null 2>&1; then
  pass "tinker CLI $(tinker version 2>/dev/null | head -1)"
else
  fail "tinker CLI missing"
fi

if command -v hf >/dev/null 2>&1; then
  pass "hf CLI present"
else
  fail "hf CLI missing"
fi

if command -v prime >/dev/null 2>&1; then
  pass "prime CLI $(prime --version 2>/dev/null | head -1 || echo present)"
else
  fail "prime CLI missing"
fi

# ── Auth: Tinker ──────────────────────────────────────────────────────────
if [[ -z "${TINKER_API_KEY:-}" ]]; then
  fail "TINKER_API_KEY unset"
else
  python - <<'PY' && pass "Tinker ServiceClient (Inkling listed)" || fail "Tinker ServiceClient"
import tinker
caps = tinker.ServiceClient().get_server_capabilities()
names = [getattr(m, "model_name", str(m)) for m in caps.supported_models]
assert any("Inkling" in n for n in names), names[:5]
print(f"       models={len(names)}")
PY
fi

# ── Auth: Hugging Face ────────────────────────────────────────────────────
if [[ -z "${HF_TOKEN:-}" ]]; then
  fail "HF_TOKEN / HUGGINGFACE_TOKEN unset"
else
  python - <<'PY' && pass "HF whoami + Inkling model_info" || fail "HF Hub"
from huggingface_hub import HfApi, model_info
api = HfApi()
who = api.whoami()
assert who.get("name"), who
info = model_info("thinkingmachines/Inkling")
print(f"       user={who.get('name')} inkling_siblings={len(info.siblings or [])}")
PY
fi

# ── Auth: Prime ───────────────────────────────────────────────────────────
if [[ -z "${PRIME_API_KEY:-}" ]]; then
  info "PRIME_API_KEY unset — skip prime API smoke"
else
  # Ensure CLI sees the key even if config.json empty
  prime --plain config set-api-key "$PRIME_API_KEY" >/dev/null
  if prime --plain availability list >/tmp/prime-avail.txt 2>&1; then
    if grep -qiE 'No API key|Error fetching' /tmp/prime-avail.txt; then
      fail "prime availability (auth)"
      head -5 /tmp/prime-avail.txt | sed 's/^/       /'
    else
      pass "prime availability list"
    fi
  else
    fail "prime availability list exited non-zero"
  fi
fi

# ── Axport R2 read ────────────────────────────────────────────────────────
echo ""
echo "=== Axport R2 read ==="
need_r2=(R2_ACCOUNT_ID R2_ACCESS_KEY_ID R2_SECRET_ACCESS_KEY)
missing_r2=0
for k in "${need_r2[@]}"; do
  if [[ -z "${!k:-}" ]]; then
    fail "$k unset"
    missing_r2=1
  fi
done

if [[ "$missing_r2" -eq 0 ]]; then
  python - <<'PY' && pass "axport R2 list+get+head" || fail "axport R2 read"
import os, json
try:
    import boto3
    from botocore.config import Config
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "boto3"])
    import boto3
    from botocore.config import Config

account = os.environ["R2_ACCOUNT_ID"]
s3 = boto3.client(
    "s3",
    endpoint_url=f"https://{account}.r2.cloudflarestorage.com",
    aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
    region_name="auto",
    config=Config(signature_version="s3v4"),
)
bucket = "axport"
listed = s3.list_objects_v2(Bucket=bucket, MaxKeys=10)
keys = [o["Key"] for o in listed.get("Contents") or []]
assert keys, "no objects in axport"
readme = next((k for k in keys if k.endswith("README.txt")), None)
if readme:
    body = s3.get_object(Bucket=bucket, Key=readme)["Body"].read()
    assert b"axport" in body.lower() or b"Bucket" in body
manifest = s3.get_object(Bucket=bucket, Key="latest/_manifest.json")["Body"].read()
data = json.loads(manifest)
assert isinstance(data, dict) and "sessions" in data or "totals" in data
head = s3.head_object(Bucket=bucket, Key="latest/export.zip")
print(f"       objects={len(keys)} manifest_bytes={len(manifest)} export_zip={head['ContentLength']}")
for k in keys[:6]:
    print(f"       - {k}")
PY
else
  info "skipping R2 object checks (missing credentials)"
fi

echo ""
if [[ "$FAIL" -eq 0 ]]; then
  echo "ALL REQUIRED CHECKS PASSED"
  exit 0
fi
echo "SOME CHECKS FAILED"
exit 1
