#!/usr/bin/env bash
# Full pipeline after data collection (step 1) and eval-set preparation. Run from the project root (Git Bash):
#     bash scripts/run_pipeline.sh            # everything
#     FROM=5 bash scripts/run_pipeline.sh     # resume from stage 5
# Logs: runs/logs/<stage>.log. Stops at the first failing stage.
#
# Data used for the reported results (October 2026):
#   uv run python scripts/01_collect_roco.py --split train      --max_findings 4000 --max_per_modality 1200
#   uv run python scripts/01_collect_roco.py --split validation --max_findings 600  --max_per_modality 200
#   eval sets: see scripts/prepare_eval_sets.py (BUSI, brain_mri_box, lung_ct_box)
set -euo pipefail
FROM=${FROM:-1}
SAM_EPOCHS=${SAM_EPOCHS:-12}
mkdir -p runs/logs

stage() {   # stage <n> <name> <command...>
    local n=$1 name=$2; shift 2
    if (( n < FROM )); then echo "skip $n $name"; return; fi
    echo "[$(date +%H:%M:%S)] stage $n: $name"
    "$@" > "runs/logs/$name.log" 2>&1 || { echo "FAILED: $name (see runs/logs/$name.log)"; exit 1; }
}

stage 1 teacher_train   uv run python scripts/02_run_teacher.py --split train
stage 2 teacher_val     uv run python scripts/02_run_teacher.py --split validation
stage 3 unet_r0         uv run python scripts/03_train_student.py --out runs/student_r0
stage 4 unet_f0         uv run python scripts/03_train_student.py --out runs/s_f0 --fold 0 --nfolds 2
stage 5 unet_f1         uv run python scripts/03_train_student.py --out runs/s_f1 --fold 1 --nfolds 2
stage 6 self_train      uv run python scripts/04_self_train.py --ckpts runs/s_f0/best.pt runs/s_f1/best.pt --reweight
stage 7 unet_r1         uv run python scripts/03_train_student.py --out runs/student_r1 \
                            --train_manifest data/pseudo/manifest_train_r1.csv
stage 8 sam_lora_r1     uv run python scripts/03b_train_sam_lora.py --out runs/sam_lora_r1 --epochs "$SAM_EPOCHS" \
                            --train_manifest data/pseudo/manifest_train_r1.csv
stage 9 eval_unet_r0    uv run python scripts/05_evaluate.py --ckpt runs/student_r0/best.pt --tta --save_vis
stage 10 eval_unet_r1   uv run python scripts/05_evaluate.py --ckpt runs/student_r1/best.pt --tta --save_vis --teacher
stage 11 eval_sam_r1    uv run python scripts/05_evaluate.py --ckpt runs/sam_lora_r1/best.pt --tta --save_vis
# the presence gate (threshold chosen on ROCO validation) does not transfer to BUSI; also score the segmenter alone
stage 12 eval_sam_gate_off uv run python scripts/05_evaluate.py --ckpt runs/sam_lora_r1/best.pt --tta --save_vis \
                            --presence_thr 0 --tag gate_off
echo "[$(date +%H:%M:%S)] done"
