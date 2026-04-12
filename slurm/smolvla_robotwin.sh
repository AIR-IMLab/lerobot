#!/usr/bin/env bash
#SBATCH --job-name=smolvla_robotwin
#SBATCH --partition=hopper-prod
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=8
#SBATCH --container-image=/fsx/pepijn/docker_images/huggingface+lerobot-gpu+dev.sqsh
#SBATCH --container-mounts=/fsx
#SBATCH --container-workdir=/admin/home/pepijn/lerobot
#SBATCH --output=/admin/home/pepijn/lerobot/slurm/logs/%x-%j.out
#SBATCH --error=/admin/home/pepijn/lerobot/slurm/logs/%x-%j.err

set -euo pipefail
source /admin/home/pepijn/lerobot/slurm/benchmark_worktree.sh

REPO=/admin/home/pepijn/lerobot
WT=/fsx/pepijn/lerobot_wt/robotwin
BRANCH=feat/robotwin-benchmark
benchmark_worktree_prepare "$REPO" "$WT" "$BRANCH"

source /admin/home/pepijn/miniconda3/etc/profile.d/conda.sh
conda activate lerobot
export_hf_lerobot_cache
cd "$WT"
export PYTHONPATH="${WT}/src${PYTHONPATH:+:${PYTHONPATH}}"

prefetch_hf_dataset "pepijn223/robotwin_unified_v3"

accelerate launch --num_processes=8 \
  -m lerobot.scripts.lerobot_train \
  --policy.path=lerobot/smolvla_base \
  --policy.repo_id=pepijn223/smolvla_robotwin \
  --policy.load_vlm_weights=true \
  --policy.scheduler_decay_steps=20000 \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --dataset.repo_id=pepijn223/robotwin_unified_v3 \
  --dataset.revision=main \
  --dataset.use_imagenet_stats=false \
  --env.type=robotwin \
  '--rename_map={"observation.images.cam_high":"observation.images.camera1","observation.images.cam_left_wrist":"observation.images.camera2","observation.images.cam_right_wrist":"observation.images.camera3"}' \
  --policy.empty_cameras=0 \
  --output_dir=outputs/train/smolvla_robotwin \
  --steps=20000 \
  --batch_size=32 \
  --eval_freq=0 \
  --save_freq=20000 \
  --policy.push_to_hub=true
