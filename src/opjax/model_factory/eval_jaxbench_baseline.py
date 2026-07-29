"""Sample Inkling (base or LoRA) on JAXBench workloads and grade the result.

1. Samples a ``workload(*inputs)`` candidate from Tinker.
2. Grades **correctness on CPU JAX** against the baseline reference.
3. Applies the anti-mimicry credit rules in :mod:`jaxbench_scoring`.
4. Shrinks large CONFIG dims for laptop-safe execution (documented in report).

Two caveats on the CPU pass, both measured (see
``docs/model-factory/06-env-rl/jaxbench-base-vs-lora.md``): it cannot score
Pallas at all (shrunken dims trip hand-written tile asserts, and the local JAX
may lag the TPU's Pallas API), and its timings are meaningless. The official
grade is ``python -m JAXBench evaluate --tpu v5e``.

Prompt context matters for scoring: ``--prompt-context spec`` (default) withholds
the reference implementation so a score cannot be earned by copying;
``--prompt-context baseline`` shows it and is therefore **diagnostic only**.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from opjax.model_factory.jaxbench_scoring import (
    COPY_SIMILARITY_THRESHOLD,
    KernelVerdict,
    PromptContext,
    judge,
    reward,
    summarise,
)

DEFAULT_SAMPLER = (
    "tinker://21e391ab-7c5d-573c-9477-16c93df81a08:train:0/sampler_weights/final"
)
DEFAULT_MODEL = "thinkingmachines/Inkling"
DEFAULT_JAXBENCH_ROOT = "/tmp/accelerator-agents"

SYSTEM_SOFT = (
    "You are an expert JAX/Pallas kernel engineer. "
    "Reply with a single Python code block that exports exactly one function:\n"
    "  def workload(*inputs): ...\n"
    "Prefer a correct Pallas (jax.experimental.pallas) kernel when it helps; "
    "pure JAX is acceptable if correct. No prose, no tests, no markdown outside "
    "the code fence."
)

SYSTEM_PALLAS_REQUIRED = (
    "You are an expert TPU Pallas kernel engineer.\n"
    "Reply with a single Python code block only. No prose.\n"
    "Hard requirements (non-negotiable):\n"
    "1. Export exactly one entrypoint: `def workload(*inputs): ...` matching the "
    "baseline argument order and output structure.\n"
    "2. The implementation MUST use `jax.experimental.pallas` via "
    "`pl.pallas_call` (and `pltpu` / scratch / BlockSpec when needed).\n"
    "3. Do NOT return a pure `jnp` / XLA rewrite. If you cannot write a real "
    "Pallas kernel, still emit a Pallas skeleton that calls `pl.pallas_call` "
    "rather than falling back to dense JAX.\n"
    "Required imports pattern:\n"
    "  from jax.experimental import pallas as pl\n"
    "  from jax.experimental.pallas import tpu as pltpu\n"
    "Typical shape:\n"
    "  def _kernel(...): ...\n"
    "  def workload(*inputs):\n"
    "      return pl.pallas_call(_kernel, out_shape=..., grid=..., "
    "in_specs=..., out_specs=...)(*inputs)\n"
)

SYSTEM = SYSTEM_SOFT  # backward-compatible default alias

# Format contract applied identically to every arm so the comparison measures
# kernel quality rather than who happens to match the extractor.
ANSWER_CONTRACT = (
    "\nOutput contract (identical for every model under test):\n"
    "- You may reason first, but keep it brief.\n"
    "- Your reply MUST end with exactly one fenced block:\n"
    "  ```python\n  <complete module defining def workload(...)>\n  ```\n"
    "- The block must be self-contained and syntactically valid Python. "
    "Close the fence."
)

RETRY_NUDGE = (
    "Your previous reply did not yield a usable code block "
    "({reason}). Reply again with NO reasoning: emit only one closed "
    "```python fenced block containing a complete, syntactically valid module "
    "that defines `workload`."
)


_SAMPLE_JUNK = re.compile(
    r"<\|(?:end_message|content_model_end_sampling|eot_id|end_of_text)\|>"
)


def _parses(src: str) -> bool:
    import ast

    try:
        ast.parse(src)
    except SyntaxError:
        return False
    return True


def _extract_code(text: str) -> str | None:
    """Return the best candidate module from a completion.

    Candidates are ranked: syntactically valid + defines ``workload`` first, so
    a rambling completion that still ends in a good block is not penalised, and
    prose is never silently handed to the grader as if it were code.
    """
    text = _SAMPLE_JUNK.sub("", text)
    candidates: list[str] = []
    candidates += re.findall(r"```(?:python|py)?\n(.*?)```", text, re.DOTALL)
    # Unclosed final fence (model ran out of budget mid-block).
    tail = re.split(r"```(?:python|py)?\n", text)
    if len(tail) > 1:
        candidates.append(tail[-1])
    if "def workload" in text:
        candidates.append(text[text.find("def workload") :])

    cleaned = [_SAMPLE_JUNK.sub("", c).strip() + "\n" for c in candidates if c.strip()]
    for want_parse in (True, False):
        for c in cleaned:
            if "def workload" not in c:
                continue
            if want_parse and not _parses(c):
                continue
            return c
    return None


def _reject_reason(code: str | None) -> str | None:
    if not code:
        return "no fenced python block containing `def workload`"
    if not _parses(code):
        return "extracted block was not valid Python (likely truncated)"
    return None


def _spec_only(baseline_src: str) -> str:
    """Baseline reduced to a spec: CONFIG, input factory, signature + docstring.

    Handing a model the reference implementation lets it pass by copying, which
    makes the score unable to distinguish kernel skill from mimicry. This keeps
    everything needed to match the I/O contract and drops the algorithm body.
    """
    import ast

    tree = ast.parse(baseline_src)
    parts: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            parts.append(ast.get_source_segment(baseline_src, node) or "")
        elif isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id.isupper() for t in node.targets
        ):
            parts.append(ast.get_source_segment(baseline_src, node) or "")
        elif isinstance(node, ast.FunctionDef) and node.name == "create_inputs":
            parts.append(ast.get_source_segment(baseline_src, node) or "")
        elif isinstance(node, ast.FunctionDef) and node.name == "workload":
            args = ast.unparse(node.args)
            doc = ast.get_docstring(node)
            body = f'    """{doc}"""\n' if doc else ""
            parts.append(f"def workload({args}):\n{body}    ...  # <- implement this")
    return "\n\n".join(p for p in parts if p) + "\n"


def _verdict_from_row(row: dict, prompt_context: str) -> KernelVerdict:
    """Rehydrate a verdict from a stored row without re-reading sources."""
    return KernelVerdict(
        workload=row.get("workload", "?"),
        correct=bool(row.get("correct")),
        uses_pallas=bool(row.get("uses_pallas")),
        prompt_context=PromptContext(row.get("prompt_context") or prompt_context),
        similarity=row.get("similarity_to_baseline"),
        verbatim_file_copy=bool(row.get("verbatim_file_copy")),
        speedup=row.get("speedup"),
        copied=bool(row.get("copied")),
        credited=bool(row.get("credited")),
        pallas_credited=bool(row.get("pallas_credited")),
        no_credit_reasons=tuple(row.get("no_credit_reasons") or ()),
    )


def _weights_path(sampler_path: str) -> str:
    if "/sampler_weights/" in sampler_path:
        return sampler_path.replace("/sampler_weights/", "/weights/", 1)
    return sampler_path


def _shrink_config(mod, max_dim: int = 128) -> dict:
    """Scale integer CONFIG entries so CPU can run create_inputs."""
    cfg = getattr(mod, "CONFIG", None)
    if not isinstance(cfg, dict):
        return {}
    ints = [int(v) for v in cfg.values() if isinstance(v, int) and v > 0]
    if not ints:
        return {}
    largest = max(ints)
    if largest <= max_dim:
        return {"scaled": False, "max_dim_used": largest}
    scale = max_dim / float(largest)
    for k, v in list(cfg.items()):
        if isinstance(v, int) and v > 1:
            cfg[k] = max(2, int(round(v * scale)))
    return {"scaled": True, "scale": scale, "max_dim_cap": max_dim}


def _grade_kernel(
    *,
    jaxbench_root: Path,
    workload_name: str,
    kernel_code: str,
    max_dim: int,
    run_timeout_s: float,
) -> dict:
    sys.path.insert(0, str(jaxbench_root))
    import jax
    import jax.numpy as jnp
    from JAXBench.benchmark import get_workload_dir
    from JAXBench.harness.correctness import check_correctness
    from JAXBench.harness.loader import load_module

    out: dict = {
        "workload": workload_name,
        "has_workload_fn": "def workload" in kernel_code,
        # Strict: an actual kernel launch, not a mention of Pallas in a comment
        # or an unused import.
        "uses_pallas": bool(re.search(r"\bpallas_call\s*\(", kernel_code)),
        "uses_jax": "jax" in kernel_code or "jnp." in kernel_code,
        "correct": False,
        "status": "error",
    }
    if not out["has_workload_fn"]:
        out["status"] = "no_workload_fn"
        out["reason"] = "generated code missing def workload"
        return out

    workload_dir = get_workload_dir(workload_name)
    baseline_path = os.path.join(workload_dir, "baseline.py")

    with tempfile.TemporaryDirectory(prefix="jb-base-") as tmp:
        kpath = Path(tmp) / "kernel.py"
        kpath.write_text(kernel_code)
        try:
            baseline_mod = load_module(baseline_path, f"{workload_name}.baseline")
            shrink = _shrink_config(baseline_mod, max_dim=max_dim)
            out["config_shrink"] = shrink
            create_fn = baseline_mod.create_inputs
            if "dtype" in create_fn.__code__.co_varnames:
                # float32 more portable on CPU than bf16 for some ops
                try:
                    inputs = create_fn(dtype=jnp.float32)
                except Exception:
                    inputs = create_fn(dtype=jnp.bfloat16)
            else:
                inputs = create_fn()
            if not isinstance(inputs, (list, tuple)):
                inputs = (inputs,)

            ref = baseline_mod.workload(*inputs)
            if hasattr(ref, "block_until_ready"):
                ref.block_until_ready()
            elif isinstance(ref, (tuple, list)):
                for x in ref:
                    if hasattr(x, "block_until_ready"):
                        x.block_until_ready()

            kernel_mod = load_module(str(kpath), f"{workload_name}.kernel")
            if not hasattr(kernel_mod, "workload"):
                out["status"] = "no_workload_fn"
                out["reason"] = "module loaded but no workload attribute"
                return out

            t0 = time.perf_counter()
            # Soft timeout via alarm not portable on all platforms; wall check after.
            test = kernel_mod.workload(*inputs)
            if hasattr(test, "block_until_ready"):
                test.block_until_ready()
            elif isinstance(test, (tuple, list)):
                for x in test:
                    if hasattr(x, "block_until_ready"):
                        x.block_until_ready()
            elapsed = time.perf_counter() - t0
            out["run_wall_s"] = round(elapsed, 3)
            if elapsed > run_timeout_s:
                out["status"] = "timeout"
                out["reason"] = f"ran {elapsed:.1f}s > {run_timeout_s}"
                return out

            chk = check_correctness(ref, test)
            out.update(chk)
            out["status"] = "correct" if chk.get("correct") else "incorrect"
            return out
        except Exception as exc:
            out["status"] = "runtime_error"
            out["reason"] = f"{type(exc).__name__}: {exc}"
            out["traceback_tail"] = traceback.format_exc()[-800:]
            return out


async def _sample(sampling, renderer, system: str, prompt: str, *, seed: int, max_tokens: int):
    from tinker import types

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    model_input = renderer.build_generation_prompt(messages)
    kwargs: dict = {}
    try:
        stops = renderer.get_stop_sequences()
        if stops:
            kwargs["stop"] = stops
    except Exception:
        pass
    params = types.SamplingParams(
        max_tokens=max_tokens,
        temperature=0.2,
        top_p=0.95,
        seed=seed,
        **kwargs,
    )
    result = await sampling.sample_async(
        prompt=model_input, num_samples=1, sampling_params=params
    )
    seq = result.sequences[0]
    return {
        "text": renderer.tokenizer.decode(seq.tokens),
        "n_tokens": len(seq.tokens),
        "stop_reason": str(getattr(seq, "stop_reason", "")),
    }


async def _sample_with_retries(
    sampling,
    renderer,
    system: str,
    prompt: str,
    *,
    seed: int,
    max_tokens: int,
    max_retries: int,
    timeout_s: float,
) -> dict:
    """Sample until a usable code block appears, or attempts run out.

    Applied identically to every arm: a model is never scored on a completion
    that was cut off mid-block without being given a chance to answer short.
    """
    attempts: list[dict] = []
    cur_prompt = prompt
    for attempt in range(max_retries + 1):
        got = await asyncio.wait_for(
            _sample(
                sampling,
                renderer,
                system,
                cur_prompt,
                seed=seed + 1000 * attempt,
                max_tokens=max_tokens,
            ),
            timeout=timeout_s,
        )
        code = _extract_code(got["text"]) or ""
        reason = _reject_reason(code or None)
        attempts.append(
            {
                "attempt": attempt,
                "n_tokens": got["n_tokens"],
                "stop_reason": got["stop_reason"],
                "truncated": got["n_tokens"] >= max_tokens,
                "completion_chars": len(got["text"]),
                "reject_reason": reason,
            }
        )
        if reason is None:
            return {"code": code, "completion": got["text"], "attempts": attempts}
        if attempt == max_retries:
            return {"code": code, "completion": got["text"], "attempts": attempts}
        cur_prompt = prompt + "\n\n" + RETRY_NUDGE.format(reason=reason)
    raise AssertionError("unreachable")


async def async_main(args: argparse.Namespace) -> int:
    import tinker
    from tinker_cookbook import model_info, renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    jaxbench_root = Path(args.jaxbench_root)
    sys.path.insert(0, str(jaxbench_root))
    from JAXBench.benchmark import list_workloads, has_optimized, get_workload_dir

    if args.prompt_context == "baseline":
        print(
            "WARNING: --prompt-context baseline shows the reference "
            "implementation. This run is DIAGNOSTIC ONLY: candidates scoring "
            f"similarity >= {COPY_SIMILARITY_THRESHOLD} against it are gated out "
            "of credit and earn zero reward.",
            flush=True,
        )

    workloads = list_workloads()
    if args.tier == "priority":
        workloads = [w for w in workloads if re.match(r"^\d+p_", w)]
    elif args.tier == "kernelbench":
        workloads = [w for w in workloads if re.match(r"^\d+k_", w)]
    if args.limit:
        workloads = workloads[: args.limit]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    kernels_dir = out_dir / "kernels"
    kernels_dir.mkdir(exist_ok=True)

    service = tinker.ServiceClient()
    t0 = time.perf_counter()
    if args.arm == "base" or not args.sampler_path:
        print(f"sampling base model {args.model_name}", flush=True)
        sampling = await service.create_sampling_client_async(base_model=args.model_name)
        arm_label = "base"
        sampler_label = None
    else:
        weights = _weights_path(args.sampler_path)
        print(f"rematerializing LoRA from {weights}", flush=True)
        training = await service.create_training_client_from_state_async(weights)
        if hasattr(training, "save_weights_and_get_sampling_client_async"):
            sampling = await training.save_weights_and_get_sampling_client_async()
        else:
            sampling = await asyncio.to_thread(training.save_weights_and_get_sampling_client)
        arm_label = "lora"
        sampler_label = args.sampler_path
    print(f"sampling client ready in {time.perf_counter() - t0:.1f}s", flush=True)

    renderer_name = model_info.get_recommended_renderer_name(args.model_name)
    tok = get_tokenizer(args.model_name)
    renderer = renderers.get_renderer(renderer_name, tok)

    if args.prompt_mode == "pallas_required":
        system = SYSTEM_PALLAS_REQUIRED + ANSWER_CONTRACT
        user_tail = (
            "Rewrite `workload` as a real TPU Pallas kernel using "
            "`pl.pallas_call` (+ `pltpu` if useful). Same I/O as baseline. "
            "Pure jnp.dot / jnp.* pipelines are REJECTED — they must not appear "
            "as the final implementation. Tile with BlockSpec / grid."
        )
        methodology_extra = " Prompt mode=pallas_required (hard Pallas)."
    else:
        system = SYSTEM_SOFT + ANSWER_CONTRACT
        if args.prompt_context == "spec":
            user_tail = (
                "Implement `workload` from the spec above. Keep the given "
                "argument order and return the documented output structure. "
                "Make it as fast as you can on TPU."
            )
        else:
            user_tail = (
                "Write an improved `workload(*inputs)` implementation. "
                "Keep the same argument order and output structure as the baseline."
            )
        methodology_extra = " Prompt mode=soft (Pallas preferred)."

    results: list[dict] = []
    done_names: set[str] = set()
    partial_path = out_dir / "partial_results.json"
    if args.resume and partial_path.exists():
        try:
            prior = json.loads(partial_path.read_text())
            if isinstance(prior, list):
                results = prior
                done_names = {r["workload"] for r in results if r.get("workload")}
                print(f"resume: loaded {len(done_names)} prior rows from {partial_path}", flush=True)
        except Exception as exc:
            print(f"resume: ignore bad partial ({exc})", flush=True)

    for i, name in enumerate(workloads, start=1):
        if name in done_names:
            print(f"[{i}/{len(workloads)}] skip {name} (resume)", flush=True)
            continue
        baseline_path = Path(get_workload_dir(name)) / "baseline.py"
        baseline_src = baseline_path.read_text()
        if args.prompt_context == "spec":
            context_src = _spec_only(baseline_src)
            context_label = (
                "Spec only — CONFIG, input factory, and the signature/docstring "
                "you must implement (no reference implementation is provided):"
            )
        else:
            context_src = baseline_src
            if len(context_src) > 12000:
                context_src = context_src[:12000] + "\n# ... truncated ...\n"
            context_label = "Baseline (must match this I/O; optimize the body):"
        prompt = (
            f"JAXBench workload: {name}\n"
            f"Has hand-optimized Pallas reference in suite: {has_optimized(name)}\n\n"
            f"{context_label}\n"
            f"```python\n{context_src}\n```\n\n"
            f"{user_tail}"
        )
        print(f"[{i}/{len(workloads)}] sample {name} …", flush=True)
        row: dict = {
            "workload": name,
            "has_optimized_ref": has_optimized(name),
            "tier": "priority" if re.match(r"^\d+p_", name) else "kernelbench_l2",
            "prompt_mode": args.prompt_mode,
            "prompt_context": args.prompt_context,
        }
        try:
            got = await _sample_with_retries(
                sampling,
                renderer,
                system,
                prompt,
                seed=args.seed + i,
                max_tokens=args.max_tokens,
                max_retries=args.max_retries,
                timeout_s=args.sample_timeout_s,
            )
            completion, code = got["completion"], got["code"]
            row["_code"] = code
            row["completion_chars"] = len(completion)
            row["code_chars"] = len(code)
            row["sample_attempts"] = got["attempts"]
            row["n_attempts"] = len(got["attempts"])
            row["truncated_any"] = any(a["truncated"] for a in got["attempts"])
            (kernels_dir / f"{name}.py").write_text(code or completion)
            if not code:
                row["status"] = "no_code"
                row["correct"] = False
            elif not _parses(code):
                row["status"] = "unparseable_code"
                row["correct"] = False
                row["reason"] = "extracted block failed ast.parse after retries"
            else:
                grade = _grade_kernel(
                    jaxbench_root=jaxbench_root,
                    workload_name=name,
                    kernel_code=code,
                    max_dim=args.max_dim,
                    run_timeout_s=args.run_timeout_s,
                )
                row.update(grade)
        except Exception as exc:
            row["status"] = "sample_error"
            row["correct"] = False
            row["reason"] = f"{type(exc).__name__}: {exc}"

        verdict = judge(
            workload=name,
            candidate_src=row.get("_code", "") or "",
            baseline_src=baseline_src,
            correct=bool(row.get("correct")),
            uses_pallas=bool(row.get("uses_pallas")),
            prompt_context=args.prompt_context,
            speedup=None,  # CPU pass cannot time; TPU merge fills this in
        )
        row.pop("_code", None)
        row["similarity_to_baseline"] = verdict.similarity
        row["verbatim_file_copy"] = verdict.verbatim_file_copy
        row["copied"] = verdict.copied
        row["credited"] = verdict.credited
        row["pallas_credited"] = verdict.pallas_credited
        row["no_credit_reasons"] = list(verdict.no_credit_reasons)
        row["reward"] = reward(verdict)
        if verdict.copied:
            row["status"] = "copied_reference"
        print(
            f"[{i}/{len(workloads)}] {name} status={row.get('status')} "
            f"correct={row.get('correct')} pallas={row.get('uses_pallas')} "
            f"attempts={row.get('n_attempts')} trunc={row.get('truncated_any')}",
            flush=True,
        )
        results.append(row)
        (out_dir / "partial_results.json").write_text(json.dumps(results, indent=2) + "\n")

    n = len(results)
    n_correct = sum(1 for r in results if r.get("correct"))
    n_pallas = sum(1 for r in results if r.get("uses_pallas"))
    n_workload = sum(1 for r in results if r.get("has_workload_fn"))
    n_truncated = sum(1 for r in results if r.get("truncated_any"))
    n_retried = sum(1 for r in results if (r.get("n_attempts") or 1) > 1)
    # Rebuilt from rows rather than accumulated in the loop, so `--resume` runs
    # score identically to single-shot runs.
    verdicts = [_verdict_from_row(r, args.prompt_context) for r in results]
    credit = summarise(verdicts)
    n_pallas_correct = credit["n_pallas_credited"]
    by_status: dict[str, int] = {}
    for r in results:
        by_status[r.get("status", "?")] = by_status.get(r.get("status", "?"), 0) + 1
    priority = [r for r in results if r.get("tier") == "priority"]
    kbench = [r for r in results if r.get("tier") == "kernelbench_l2"]

    summary = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "model_name": args.model_name,
        "arm": arm_label,
        "sampler_path": sampler_label,
        "jaxbench_root": str(jaxbench_root),
        "prompt_mode": args.prompt_mode,
        "prompt_context": args.prompt_context,
        "methodology": (
            f"Tinker sample arm={arm_label}; CPU JAX correctness vs baseline; "
            f"CONFIG ints capped via shrink to max_dim={args.max_dim}. "
            "Not official TPU JAXBench timing (and CPU cannot score Pallas). "
            f"prompt_context={args.prompt_context}; copy gate "
            f"{'ACTIVE' if PromptContext(args.prompt_context).gates_copies else 'inactive (no reference shown)'} "
            f"at similarity >= {COPY_SIMILARITY_THRESHOLD}."
            + methodology_extra
        ),
        "n": n,
        "n_correct": n_correct,
        "pass_rate": (n_correct / n) if n else 0.0,
        "n_has_workload_fn": n_workload,
        "n_uses_pallas": n_pallas,
        "n_pallas_correct": n_pallas_correct,
        "credit": credit,
        "copy_similarity_threshold": COPY_SIMILARITY_THRESHOLD,
        "scorable": credit["scorable"],
        "scorable_note": (
            "prompt_context=spec: reference withheld, score reflects skill"
            if credit["scorable"]
            else "prompt_context=baseline: reference shown, DIAGNOSTIC ONLY — "
            "near-copies of the reference are gated out of credit"
        ),
        "max_tokens": args.max_tokens,
        "max_retries": args.max_retries,
        "n_truncated_any_attempt": n_truncated,
        "n_needed_retry": n_retried,
        "by_status": by_status,
        "priority_pass_rate": (
            sum(1 for r in priority if r.get("correct")) / len(priority) if priority else None
        ),
        "kernelbench_l2_pass_rate": (
            sum(1 for r in kbench if r.get("correct")) / len(kbench) if kbench else None
        ),
        "results": results,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    md = [
        f"# JAXBench baseline — Inkling `{arm_label}` (CPU functional)",
        "",
        f"**When:** `{summary['checked_at']}`",
        f"**Arm:** `{arm_label}`",
        f"**Sampler:** `{sampler_label}`",
        f"**Prompt context:** `{args.prompt_context}` — {summary['scorable_note']}",
        f"**Credited:** **{credit['n_credited']}/{n}** "
        f"(raw correct {n_correct}, copies gated {credit['n_copied']} "
        f"of which {credit['n_verbatim_file_copies']} verbatim files)",
        f"**Mean reward:** {credit['mean_reward']}",
        f"**Pass rate (raw, ungated):** {summary['pass_rate']:.3f} ({n_correct}/{n})",
        f"**Priority (1p–17p):** {summary['priority_pass_rate']}",
        f"**KernelBench L2 (18k–50k):** {summary['kernelbench_l2_pass_rate']}",
        f"**Emits `workload`:** {n_workload}/{n} · **Mentions Pallas:** {n_pallas}/{n}"
        f" · **pallas_correct:** {n_pallas_correct}/{n}",
        f"**Budget:** max_tokens={args.max_tokens}, max_retries={args.max_retries} · "
        f"truncated on some attempt: {n_truncated}/{n} · needed retry: {n_retried}/{n}",
        "",
        "## Methodology",
        "",
        summary["methodology"],
        "",
        "## By status",
        "",
        "```json",
        json.dumps(by_status, indent=2),
        "```",
        "",
        "## Per workload",
        "",
        "| Workload | Tier | Status | Correct | Pallas | Sim | Credited |",
        "|----------|------|--------|---------|--------|-----|----------|",
    ]
    for r in results:
        sim = r.get("similarity_to_baseline")
        md.append(
            f"| `{r['workload']}` | {r.get('tier')} | {r.get('status')} | "
            f"{r.get('correct')} | {r.get('uses_pallas')} | "
            f"{'—' if sim is None else f'{sim:.2f}'} | {r.get('credited')} |"
        )
    md.append("")
    (out_dir / "REPORT.md").write_text("\n".join(md))
    print(json.dumps({k: summary[k] for k in summary if k != "results"}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="JAXBench CPU baseline for Stage-5 LoRA")
    p.add_argument("--sampler-path", default=DEFAULT_SAMPLER)
    p.add_argument(
        "--arm",
        choices=["lora", "base"],
        default="lora",
        help="lora=Stage-5 sampler; base=thinkingmachines/Inkling with no adapter",
    )
    p.add_argument("--model-name", default=DEFAULT_MODEL)
    p.add_argument("--jaxbench-root", default=DEFAULT_JAXBENCH_ROOT)
    p.add_argument("--out-dir", default="data/model-factory/evals/jaxbench-baseline-lora")
    p.add_argument("--tier", choices=["all", "priority", "kernelbench"], default="all")
    p.add_argument(
        "--prompt-mode",
        choices=["soft", "pallas_required"],
        default="soft",
        help="soft=prefer Pallas; pallas_required=hard-require pl.pallas_call",
    )
    p.add_argument(
        "--prompt-context",
        choices=["spec", "baseline"],
        default="spec",
        help=(
            "spec (default, scorable)=withhold the reference implementation; "
            "baseline (diagnostic)=show it, near-copies then earn no credit"
        ),
    )
    p.add_argument("--limit", type=int, default=0, help="0 = no limit")
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip workloads already present in out-dir/partial_results.json",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Re-sample when no valid fenced `workload` block comes back",
    )
    p.add_argument("--max-dim", type=int, default=128, help="CONFIG shrink cap for CPU")
    p.add_argument("--sample-timeout-s", type=float, default=600.0)
    p.add_argument("--run-timeout-s", type=float, default=120.0)
    args = p.parse_args(argv)
    if args.limit == 0:
        args.limit = None
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
