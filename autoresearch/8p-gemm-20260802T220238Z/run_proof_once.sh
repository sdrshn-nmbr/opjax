#!/usr/bin/env bash
set -euo pipefail

output_dir=/home/sudarshan/ar8p/8p-gemm-20260802T220238Z/final_proof_verified
lock_path=/tmp/opjax-ar8p-final-proof.lock
mkdir -p "$output_dir"
exec 9>"$lock_path"
if ! flock -n 9; then
    exit 0
fi
if [[ -f "$output_dir/done" ]]; then
    exit 0
fi

cd /home/sudarshan/ar8p/accelerator-agents
set +e
PYTHONPATH=. timeout --signal=TERM --kill-after=10s 90s \
    /home/sudarshan/ar8p/pallas-eval/.venv/bin/python \
    /home/sudarshan/ar8p/8p-gemm-20260802T220238Z/final_proof.py \
    --candidate /home/sudarshan/ar8p/8p-gemm-20260802T220238Z/candidates/iter_016_bm1024_bn1024_bk512.py \
    --output-dir "$output_dir" \
    >"$output_dir/stdout.log" 2>"$output_dir/stderr.log"
return_code=$?
set -e
printf '%s\n' "$return_code" >"$output_dir/exit_code"
touch "$output_dir/done"
exit "$return_code"
