# DeepSWE usage policy

**Public site:** https://deepswe.datacurve.ai/run (~113 tasks class of benches)

## Binding rules

1. The **public report / leaderboard split** is **sealed for reporting**.
   It must **never** appear in SFT/RL train fuel used for scores we report.
2. Failures on the public report split must **not** be mined into train.
   Route hard cases to a **separate train task pool** or rotating-dev with distinct ids.
3. Pier / Harbor runners may evaluate the report split for measurement only.
4. Contaminating the report split voids public DeepSWE claims in this project.

## Practical checklist

- [ ] Train JSONL provenance excludes DeepSWE report task ids
- [ ] Eval job config records split name + commit / registry pin
- [ ] CL / iSFT loops refuse report-split task ids as fuel

## Relationship to SudarshanBench

DeepSWE = public sealed/report. SudarshanBench = private sealed.
Neither is train fuel for the split being reported.
