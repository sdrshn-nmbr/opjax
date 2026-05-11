#!/usr/bin/env node
// Copies the 23 source .md files from /opjax/references/{bstn,tm}/ into
// src/content/chapters/ with chapter-numbered filenames so Astro's content
// collection picks them up in canonical reading order.

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, '..');
const refsRoot = path.resolve(root, '..', 'references');
const dest = path.resolve(root, 'src', 'content', 'chapters');

const manifest = [
  ['01', 'michael-scott', 'bstn/posts/research/the-michael-scott-paper-company-of-ai/index.md'],
  ['02', 'continual-learning', 'bstn/posts/research/continual-learning/index.md'],
  ['03', 'lora-without-regret', 'tm/blog/lora/index.md'],
  ['04', 'iterative-sft', 'bstn/posts/research/iterative-sft/index.md'],
  ['05', 'prompt-mutations', 'bstn/posts/research/robust-sample-efficient-sft-with-prompt-mutations/index.md'],
  ['06', 'rgt', 'bstn/posts/research/upweight-the-strategy-not-the-tokens-faster-training-with-explicit-reasoning-thro/index.md'],
  ['07', 'on-policy-distillation', 'tm/blog/on-policy-distillation/index.md'],
  ['08', 'gad', 'bstn/posts/research/distillation-without-the-dark/index.md'],
  ['09', 'dense-on-policy-or-both', 'bstn/posts/research/dense-on-policy-or-both/index.md'],
  ['10', 'practical-lora', 'bstn/posts/research/practical-lora-research/index.md'],
  ['11', 'write-small-learn-forever', 'bstn/posts/research/write-small-learn-forever/index.md'],
  ['12', 'lumina', 'bstn/posts/research/lumina-building-self-improving-evaluation-through-customer-in-the-loop-refinement/index.md'],
  ['13', 'training-loss-predicts-eval', 'bstn/posts/research/training-loss-predicts-evaluation-performance-even-for-non-verifiable-tasks/index.md'],
  ['14', 'byo-swe-grep', 'bstn/posts/research/byo-swe-grep/index.md'],
  ['15', 'repeated-kv-cache', 'bstn/posts/research/repeated-kv-cache-for-long-running-agents/index.md'],
  ['16', 'still', 'bstn/posts/research/towards-infinite-context-windows-neural-kv-cache-compaction/index.md'],
  ['17', 'radixmlp', 'bstn/posts/research/introducing-radixmlp-intra-batch-deduplication-for-causal-transformers/index.md'],
  ['18', 'defeating-nondeterminism', 'tm/blog/defeating-nondeterminism-in-llm-inference/index.md'],
  ['19', 'modular-manifolds', 'tm/blog/modular-manifolds/index.md'],
  ['20', 'attention-attribution', 'bstn/posts/research/attention-based-attribution/index.md'],
  ['21', 'hallucination-probe', 'bstn/posts/research/do-transformers-notice-their-own-mistakes/index.md'],
  ['22', 'resurrecting-the-salmon', 'bstn/posts/research/resurrecting-the-salmon/index.md'],
  ['23', 'mech-interp-paradigm', 'bstn/posts/research/mechanistic-interpretability/index.md'],
];

// Two known markdown-vs-LaTeX collisions in the source posts:
//
//   1. `\tag{$\star$}` and friends embed `$...$` inside `$$...$$`, which
//      breaks remark-math's display delimiter detection. KaTeX's \tag takes
//      math content directly, so we drop the inner `$`.
//
//   2. defuddle emitted display equations as single-line `$$X$$`, but
//      remark-math only treats `$$...$$` as block (display) math when the
//      opening and closing fences are on their own lines. We expand any
//      whole-line `$$...$$` into the multi-line form.
function preprocess(md) {
  let out = md;
  out = out.replace(/\\\\tag\{\$\\\\([A-Za-z]+)\$\}/g, '\\\\tag{\\\\$1}');
  out = out.replace(/^(\s*)\$\$([^\n]+?)\$\$\s*$/gm, (m, indent, eq) => {
    if (eq.includes('\n')) return m;
    return `${indent}$$\n${eq.trim()}\n$$`;
  });
  return out;
}

fs.mkdirSync(dest, { recursive: true });

let copied = 0;
const missing = [];
for (const [n, slug, src] of manifest) {
  const from = path.join(refsRoot, src);
  const to = path.join(dest, `${n}-${slug}.md`);
  if (!fs.existsSync(from)) {
    missing.push(from);
    continue;
  }
  const raw = fs.readFileSync(from, 'utf8');
  fs.writeFileSync(to, preprocess(raw));
  console.log(`  ✓ ${n}-${slug}.md`);
  copied++;
}

if (missing.length) {
  console.error(`\nMISSING SOURCES (${missing.length}):`);
  for (const m of missing) console.error(`  ✗ ${m}`);
  process.exit(1);
}

console.log(`\nSynced ${copied} chapters → src/content/chapters/`);
