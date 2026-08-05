#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.env"

"${PYTHON_BIN}" "${SCRIPT_DIR}/check_joint_leakage.py" "${DATA_HUMAN}"

allow_cpu=()
if [[ "${ALLOW_CPU:-0}" == "1" ]]; then
  allow_cpu=(--allow-cpu)
fi

for specification in ${RUNS}; do
  variant="${specification%%:*}"
  tissue_set="${specification##*:}"
  case "${variant}:${tissue_set}" in
    baseline:with_combined|bioaware:with_combined|baseline:no_combined|bioaware:no_combined)
      ;;
    *)
      echo "Invalid RUNS entry: ${specification}" >&2
      exit 2
      ;;
  esac

  echo "[train] variant=${variant} tissue_set=${tissue_set}"
  "${PYTHON_BIN}" "${ANALYSIS_DIR}/pipeline/train_joint.py" \
    --variant "${variant}" \
    --tissue-set "${tissue_set}" \
    --data-root "${DATA_HUMAN}" \
    --cache-root "${CACHE_ROOT}" \
    --out-root "${OUT_ROOT}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --num-threads "${NUM_THREADS}" \
    --checkpoint-every "${CHECKPOINT_EVERY}" \
    --seed "${SEED}" \
    --resume \
    "${allow_cpu[@]}"
done

echo "PASS: requested joint-model runs completed under ${OUT_ROOT}"
