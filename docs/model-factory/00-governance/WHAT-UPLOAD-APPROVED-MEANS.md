# What `upload_approved` means

It is **not** “is axport usable?”

| Flag | Answers |
|------|---------|
| Data slice `rights_cleared` | May we **use this corpus** (e.g. axport) for the experiment at all? |
| Provider `upload_approved` | May we **send scrubbed training files to this company** (Tinker, Prime, Modal, …)? |

`upload_approved: false` was the **fail-closed default** I shipped so the CLI would refuse uploads until a human said which vendors may receive data.

Axport can be cleared as a **source** while Prime/Fireworks stay `upload_approved: false`. Training on Tinker only requires Tinker’s flag (plus scrub + spend headroom).
