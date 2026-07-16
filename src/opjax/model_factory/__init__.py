"""Model Factory: governance, sealed evals, and data-factory tooling.

Stages 0–2 are enforceable locally. Later stages are gated runbooks under
``docs/model-factory/`` — do not upload private traces until Stage-0 passes.
"""

from opjax.model_factory.audit import audit_jsonl, AuditReport
from opjax.model_factory.canary import CanarySet, embed_canaries, find_canaries
from opjax.model_factory.pre_upload import PreUploadResult, run_pre_upload_gate
from opjax.model_factory.scrub import ScrubResult, scrub_text
from opjax.model_factory.splits import SplitManifest, load_split_manifest, validate_no_train_on_sealed

__all__ = [
    "AuditReport",
    "CanarySet",
    "PreUploadResult",
    "ScrubResult",
    "SplitManifest",
    "audit_jsonl",
    "embed_canaries",
    "find_canaries",
    "load_split_manifest",
    "run_pre_upload_gate",
    "scrub_text",
    "validate_no_train_on_sealed",
]
