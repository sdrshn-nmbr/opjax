# DeepSWE usage policy

**Primary source:** https://deepswe.datacurve.ai/run (~113 public tasks)

## Binding rules

1. The **public leaderboard / report split** is for measurement only.
2. **Never** use failures or solutions from that report split as SFT/RL/CL fuel for any model whose DeepSWE score you will report.
3. If you need training environments, maintain a **separate** train task pool (synthetic, allowlisted OSS, or an explicitly non-report DeepSWE subset if the benchmark publishers define one).
4. List every DeepSWE task id used for reporting in `splits.json` → `deepswe_report_split`.
5. `validate_no_train_on_sealed` treats `deepswe_report_split` as forbidden for training IDs.

## Runner notes

- Pier (`datacurve-pier`) is the candidate Harbor-compatible runner.
- Harbor / Terminal-Bench are secondary and follow the same train-vs-report discipline.
