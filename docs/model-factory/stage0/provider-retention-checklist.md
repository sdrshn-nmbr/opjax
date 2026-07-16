# Provider retention / disclosure checklist (Stage 0)

Answer before approving a rights manifest that lists the provider.
Do not invent policy — link to vendor docs or mark **UNKNOWN / accepted risk**.

## Tinker (Thinking Machines)

| Question | Answer | Source |
|----------|--------|--------|
| Is training JSONL retained after the run? How long? | | Console / docs / DPA |
| Who can access run artifacts (org, support)? | | |
| Are samples logged for abuse review? | | |
| Checkpoint export / deletion procedure | | |
| Zero-retention option? | | |

Docs: https://tinker-docs.thinkingmachines.ai/ · Console: https://tinker-console.thinkingmachines.ai

## Prime Intellect

| Question | Answer | Source |
|----------|--------|--------|
| Hosted training log retention | | |
| Pod disk lifetime after stop | | |
| Env / rollout storage | | |

Note: Inkling **not** on Hosted Training allowlist — pods/BYO later.

## Hugging Face Hub

| Question | Answer | Source |
|----------|--------|--------|
| Repo visibility for adapters | private by default for this project? | |
| Gated model license acceptance for Inkling | N/A (ungated Apache-2.0 as of 2026-07-16) | Hub card |

## Modal / Together / Fireworks / Baseten

Defer until Wave B. If used for infer, fill log retention + zero-retention options.

## Anthropic / OpenAI (Pier graders)

| Question | Answer | Source |
|----------|--------|--------|
| API data use / training on API inputs | | Vendor policy |
| What task text may be sent | | |

## Cloudflare R2 (axport warehouse)

| Question | Answer | Source |
|----------|--------|--------|
| Bucket IAM least privilege | | |
| Lifecycle rules | | |
| Confirmed this is **our** bucket (not a trainer) | Yes when configured | |

## Decision rule

If retention is **UNKNOWN**, either (a) do not upload private data, or (b) record
“accepted risk” with owner name + date on the slice sign-off.
