# Stage 0 — Provider retention / disclosure checklist

Feeds the rights manifest. Answer before setting `retention_reviewed: true`.

| Provider | Data that may leave laptop | Questions to answer | Status |
|----------|----------------------------|---------------------|--------|
| **Tinker** | Training JSONL, run metadata, checkpoints | Retention of training data? Who can access runs? Subprocessors? | Wave A account exists; **retention_reviewed = false** until terms read |
| **Prime** | Envs, rollouts, pod disks, logs | Hosted training log lifetime; pod disk TTL; inference API logging | CLI logged in; review pending |
| **Modal** | Volumes, logs, secrets | Volume retention; secret scoping (`opjax-secrets`) | Profile `conway` ready; review pending |
| **Fireworks** | Prompts/completions if infer/RFT | Log retention; zero-retention options; FT data use | Deferred (Wave B) |
| **Baseten** | Same | Loops/training retention when GA | Deferred (Wave C) |
| **HF Hub** | Model/dataset repos if push | Private repo ACL; accidental public push | Login done; push policy pending |
| **R2 / axport** | Trace dumps in **our** bucket | IAM, lifecycle rules | OK if bucket stays private |
| **Anthropic / OpenAI** | Pier graders / iSFT repair | API data-use / training-on-inputs policies | Keys present; document before bulk grading |

## Rule

Auth ≠ upload. Wave A accounts may smoke **public** models. Private axport/Conway requires this checklist + signed JSON manifest + passing `pre-upload-gate`.
