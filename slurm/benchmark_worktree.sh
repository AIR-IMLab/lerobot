#!/usr/bin/env bash
# Prepare a dedicated git worktree for a benchmark branch (safe for concurrent SLURM jobs).
# Usage: benchmark_worktree_prepare REPO_DIR WORKTREE_DIR BRANCH_NAME
benchmark_worktree_prepare() {
  local repo="$1" wt="$2" branch="$3"
  mkdir -p "$(dirname "$wt")"
  git -C "$repo" fetch origin "$branch"
  if git -C "$wt" rev-parse --git-dir >/dev/null 2>&1; then
    git -C "$wt" fetch origin "$branch"
    git -C "$wt" checkout "$branch"
    if git -C "$wt" diff-index --quiet HEAD -- 2>/dev/null; then
      git -C "$wt" merge --ff-only "origin/$branch" || true
    else
      echo "Worktree ${wt} has local changes; skipping merge with origin/${branch}."
    fi
  else
    if git -C "$repo" show-ref --verify --quiet "refs/heads/$branch"; then
      git -C "$repo" worktree add "$wt" "$branch"
    else
      git -C "$repo" worktree add -b "$branch" "$wt" "origin/$branch"
    fi
  fi
}

# Match LeRobot defaults (see lerobot.utils.constants): datasets live under HF_LEROBOT_HOME / repo_id.
export_hf_lerobot_cache() {
  export HF_HOME="${HF_HOME:-/fsx/pepijn/.cache/huggingface}"
  export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${HF_HOME}/lerobot}"
}

# Snapshot a dataset into HF_LEROBOT_HOME/<repo_id> (same path training uses).
# Optional second arg: revision (default main). Use the same revision as --dataset.revision
# so LeRobot does not fall back to CODEBASE_VERSION (e.g. v3.0) and hit RevisionNotFoundError
# on repos without matching Hub tags.
prefetch_hf_dataset() {
  local repo_id="$1"
  local revision="${2:-main}"
  export_hf_lerobot_cache
  echo "Prefetching dataset ${repo_id} (revision=${revision}) into \${HF_LEROBOT_HOME}..."
  python -c "
import os
from pathlib import Path
from huggingface_hub import snapshot_download
repo_id = '${repo_id}'
revision = '${revision}'
root = Path(os.environ['HF_LEROBOT_HOME']) / repo_id
root.mkdir(parents=True, exist_ok=True)
snapshot_download(repo_id=repo_id, repo_type='dataset', local_dir=str(root), revision=revision)
"
}
