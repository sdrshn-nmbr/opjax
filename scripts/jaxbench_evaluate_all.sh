#!/usr/bin/env bash
# Sequential JAXBench evaluate — one process at a time (libtpu exclusive).
set -u
KERNELS_DIR=${1:-kernels_base}
TPU=${2:-v5e}
OUT=${3:-tpu_eval_results.jsonl}
SUMMARY=${4:-tpu_eval_summary.json}
NUM_WARMUP=${NUM_WARMUP:-3}
NUM_ITERS=${NUM_ITERS:-20}

export PYTHONPATH="${HOME}/jaxbench:${PYTHONPATH:-}"
export PATH="${HOME}/.local/bin:${PATH:-}"

rm -f "$OUT" "$SUMMARY"
shopt -s nullglob
filtered=()
for f in "$KERNELS_DIR"/*.py; do
  base=$(basename "$f")
  [[ "$base" == ._* ]] && continue
  filtered+=("$f")
done
n=${#filtered[@]}
echo "evaluate $n kernels --tpu $TPU (sequential)"
i=0
for kpath in "${filtered[@]}"; do
  i=$((i + 1))
  workload=$(basename "$kpath" .py)
  echo "=== [$i/$n] evaluating $workload ==="
  python3 -m JAXBench evaluate \
    --workload "$workload" \
    --kernel "$kpath" \
    --tpu "$TPU" \
    --num-warmup "$NUM_WARMUP" \
    --num-iters "$NUM_ITERS" \
    --json >"/tmp/jb_${workload}.out" 2>/tmp/jb_${workload}.err || true

  python3 - "$workload" "/tmp/jb_${workload}.out" "/tmp/jb_${workload}.err" "$OUT" <<'PY'
import json, sys
from pathlib import Path
workload, out_p, err_p, dest = sys.argv[1:5]
text = Path(out_p).read_text()
parsed = None
# full-file JSON
try:
    parsed = json.loads(text.strip())
except Exception:
    pass
if parsed is None:
    # find last balanced {...}
    start = text.rfind("{")
    while start >= 0 and parsed is None:
        try:
            parsed = json.loads(text[start:])
        except Exception:
            start = text.rfind("{", 0, start)
            continue
        break
if parsed is None:
    err = Path(err_p).read_text()[-500:]
    parsed = {"workload": workload, "status": "error", "correct": None, "error": err}
parsed.setdefault("workload", workload)
line = json.dumps(parsed, default=str)
Path(dest).open("a").write(line + "\n")
print(line[:400], flush=True)
PY
  sleep 1
done

python3 - <<PY
import json
from pathlib import Path
rows=[]
for line in Path("$OUT").read_text().splitlines():
    line=line.strip()
    if not line: continue
    try: rows.append(json.loads(line))
    except: pass
status={}
n_correct=0
speedups=[]
for r in rows:
    st=r.get("status","error")
    status[st]=status.get(st,0)+1
    if r.get("correct") is True:
        n_correct+=1
        sp=r.get("speedup")
        if isinstance(sp,(int,float)): speedups.append(float(sp))
speedups.sort()
summary={"n":len(rows),"correct":n_correct,"status_counts":status,
         "n_correct_with_speedup":len(speedups),
         "median_speedup":(speedups[len(speedups)//2] if speedups else None),
         "best_speedup":(speedups[-1] if speedups else None),
         "n_speedup_gt_1":sum(1 for s in speedups if s>1.0)}
Path("$SUMMARY").write_text(json.dumps(summary, indent=2)+"\n")
print(json.dumps(summary, indent=2))
PY
