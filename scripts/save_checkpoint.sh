#!/usr/bin/env bash

set -euo pipefail

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Not inside a git repository."
  exit 1
fi

branch="$(git branch --show-current)"
message="${1:-checkpoint: ${branch} $(date '+%Y-%m-%d %H:%M:%S')}"

if [[ -z "$(git status --short)" ]]; then
  echo "No changes to save."
  exit 0
fi

git add -A
git commit -m "$message"

echo "Saved checkpoint on branch '${branch}': $message"
