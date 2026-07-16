# Composer, SWE-1.7, Devin Fusion

## Primary sources

- Composer 2.5: https://cursor.com/blog/composer-2-5
- SWE-1.7: https://cognition.com/blog/swe-1-7
- Devin Fusion: https://cognition.com/blog/devin-fusion
- Local mirrors (if present): `references/composer2/`, `references/swe-1.7/` (may be machine-local)

## Transfer boundary

- Composer textual feedback = same-policy KL + local hint, not “ask a stronger model to rewrite traces.”
- SWE-1.7 starts from heavily RL’d bases and multi-cluster infra.
- Fusion: same tokenizer ≠ shared KV; role split is unproven.
