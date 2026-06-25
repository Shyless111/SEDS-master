# Branching Guide

This repository mixes baseline maintenance, method development, and many short-lived experiment runs. To keep the history readable, use a small set of branch types with stable naming.

## Branch Roles

- `main`
  - Always keep `main` in a runnable and paper-safe state.
  - Only merge code that you are willing to reproduce later.

- `dev`
  - Optional integration branch for ongoing code changes before they are stable enough for `main`.
  - Use this when several method changes are evolving together.

- `feat/<topic>`
  - For new method code or larger refactors.
  - Examples:
    - `feat/bert-text-encoder`
    - `feat/uatvr-head`
    - `feat/dual-softmax-eval`

- `fix/<topic>`
  - For bug fixes that should stay easy to locate later.
  - Examples:
    - `fix/metrics-direction`
    - `fix/eval-sim-shape`

- `exp/<dataset>-<idea>-<tag>`
  - For experiment-only code that may change fast and may not deserve a long lifetime.
  - Keep these branches short-lived.
  - Examples:
    - `exp/ph-b128-bdsl-n8`
    - `exp/ph-bert-uatvr-dsl`
    - `exp/ph-bs64-adam`

- `paper/<topic>`
  - For code cleanup or result-locking work done specifically for writing or release.
  - Examples:
    - `paper/final-ablation-table`
    - `paper/repro-check`

## Recommended Workflow

1. Start from `main` for isolated experiments:

```bash
git switch main
git pull
git switch -c exp/ph-b128-bdsl-n8
```

2. Start from `dev` when the experiment depends on unfinished method code:

```bash
git switch dev
git pull
git switch -c exp/ph-bert-uatvr-dsl
```

3. Merge only the code worth keeping:
   - Result logs stay local and are already ignored.
   - Temporary launch scripts or one-off debug prints should be cleaned before merge.

## Naming Rules

- Use lowercase only.
- Separate words with `-`.
- Put the most important retrieval setting in the branch name:
  - dataset: `ph`, `h2`, `csl`
  - encoder: `bert`, `clip`
  - method: `uatvr`, `dsl`, `fg`
  - key batch size or ablation tag: `b128`, `n8`, `adam`

Good examples:

- `exp/ph-b128-bdsl-n8`
- `feat/bert-tokenizer-adapter`
- `fix/video-to-text-metric`

Avoid:

- `test`
- `newcode`
- `try1`
- `final_final2`

## Commit Message Style

Keep commit messages short and searchable:

```text
feat: add distilbert text encoder path
feat: integrate uatvr uncertainty head
fix: correct retrieval metric direction
exp: tune probabilistic embedding count to 8
paper: freeze bs128 e500 ablation setup
```

## Quick Checkpoints

When you want to save the current code state before another experiment or refactor, use:

```bash
bash scripts/save_checkpoint.sh
```

Or add a custom message:

```bash
bash scripts/save_checkpoint.sh "exp: adjust dsl evaluation path"
```

This is useful for frequent research iterations when you want to switch branches or compare code states later.

## What Should Reach `main`

Good candidates:

- Stable training or evaluation code
- Reproducible ablation switches
- Clean script updates
- Metric fixes
- Documentation for experiments

Keep out of `main`:

- Half-finished debugging code
- Branch-specific hacks for one run
- Local machine paths unless they are configurable
- Raw logs, checkpoints, and datasets

## Suggested Branch Layout For This Project

If you want a simple long-term structure, use:

- `main`: stable default branch
- `dev`: current development line
- `feat/*`: method implementation
- `exp/*`: temporary experiment branches
- `paper/*`: final result consolidation

This is enough for a research codebase without becoming heavy.
