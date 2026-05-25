#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CONDA_ENV="${CONDA_ENV:-libero}"
SUITE="${SUITE:-libero_10}"
GPU_ID="${GPU_ID:-7}"
CHECKPOINT="${CHECKPOINT:-openvla/openvla-7b-finetuned-libero-10}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
NUM_STEPS_WAIT="${NUM_STEPS_WAIT:-10}"
AUGMENTATION_TIMES="${AUGMENTATION_TIMES:-2}"
PERTURB_SCALE="${PERTURB_SCALE:-1.0}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
KL_COEF="${KL_COEF:-0.01}"
FEEDBACK_BASELINE="${FEEDBACK_BASELINE:-0.5}"
BASELINE_MOMENTUM="${BASELINE_MOMENTUM:-0.9}"
UPDATE_BATCH_SIZE="${UPDATE_BATCH_SIZE:-64}"
UPDATE_EPOCHS="${UPDATE_EPOCHS:-1}"
SAVE_VIDEOS="${SAVE_VIDEOS:-False}"
MUJOCO_GL="${MUJOCO_GL:-glx}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"

LOG_ROOT="${LOG_ROOT:-${ROOT_DIR}/results/pdf-${SUITE}-${RUN_ID}}"
mkdir -p "${LOG_ROOT}/logs"

SUMMARY_FILE="${LOG_ROOT}/summary.tsv"
printf "suite\ttask_id\tcheckpoint\texit_code\tstatus\tlog_file\tfinal_summary\n" > "${SUMMARY_FILE}"

TASK_IDS=(0 1 2 3 4 5 6 7 8 9)

echo "PDF 10-task run"
echo "  root: ${ROOT_DIR}"
echo "  conda env: ${CONDA_ENV}"
echo "  suite: ${SUITE}"
echo "  gpu: ${GPU_ID}"
echo "  checkpoint: ${CHECKPOINT}"
echo "  logs: ${LOG_ROOT}"
echo "  trials/task: ${NUM_TRIALS_PER_TASK}"
echo "  save videos: ${SAVE_VIDEOS}"

for task_id in "${TASK_IDS[@]}"; do
  log_file="${LOG_ROOT}/logs/${SUITE}-task-${task_id}.log"
  echo "[$(date '+%F %T')] start suite=${SUITE} task=${task_id}" | tee "${log_file}"

  CUDA_VISIBLE_DEVICES="${GPU_ID}" MUJOCO_GL="${MUJOCO_GL}" \
  xvfb-run -s "-screen 0 1280x720x24" -a \
  conda run --no-capture-output -n "${CONDA_ENV}" \
    python -m pdflibero.pdf \
      --model_family openvla \
      --pretrained_checkpoint "${CHECKPOINT}" \
      --task_suite_name "${SUITE}" \
      --task_id "${task_id}" \
      --center_crop True \
      --num_trials_per_task "${NUM_TRIALS_PER_TASK}" \
      --num_steps_wait "${NUM_STEPS_WAIT}" \
      --augmentation_times "${AUGMENTATION_TIMES}" \
      --perturb_scale "${PERTURB_SCALE}" \
      --learning_rate "${LEARNING_RATE}" \
      --kl_coef "${KL_COEF}" \
      --feedback_baseline "${FEEDBACK_BASELINE}" \
      --baseline_momentum "${BASELINE_MOMENTUM}" \
      --update_batch_size "${UPDATE_BATCH_SIZE}" \
      --update_epochs "${UPDATE_EPOCHS}" \
      --save_videos "${SAVE_VIDEOS}" \
    2>&1 | tee -a "${log_file}"

  exit_code=${PIPESTATUS[0]}
  status="ok"
  if [[ "${exit_code}" -ne 0 ]]; then
    status="failed"
  fi

  final_summary="$(grep -F "Final success:" "${log_file}" | tail -n 1 | tr '\t' ' ' || true)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${SUITE}" "${task_id}" "${CHECKPOINT}" "${exit_code}" "${status}" "${log_file}" "${final_summary}" \
    >> "${SUMMARY_FILE}"

  echo "[$(date '+%F %T')] done suite=${SUITE} task=${task_id} status=${status} exit=${exit_code}"
done

echo "All 10 ${SUITE} experiments finished. Summary: ${SUMMARY_FILE}"
