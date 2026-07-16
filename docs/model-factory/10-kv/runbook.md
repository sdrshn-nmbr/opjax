# Stage 10 — KV

## 10a Engineering (ship path)

Same-checkpoint only:

- Prefix cache
- TTL
- Sticky sessions
- Text compaction

## 10b Research (after Fusion has measurable context cost)

- Latent Briefing: worker re-encodes orchestrator **text** into worker-native KV — not cross-model cache handoff.
- STILL / neural compaction: model-specific; factual evidence so far.

## Forbidden

- Assuming shared tokenizer ⇒ interchangeable KV.
- Promoting continually updated sidekick without version pins (invalidates caches / in-flight sessions).
