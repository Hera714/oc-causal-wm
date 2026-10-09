#!/usr/bin/env bash
# Data preparation only. Run on AutoDL CPU; no GPU is needed.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PREP_PYTHON="${PREP_PYTHON:-/root/autodl-tmp/venvs/movi-prep/bin/python}"
DATA_ROOT="${DATA_ROOT:-/autodl-fs/data/movi_c_flow_v2_1200}"
STAGING_ROOT="${STAGING_ROOT:-/root/autodl-tmp/movi_c_1200_stage}"
SEED="${SEED:-20261008}"

[[ -f "$SCRIPT_DIR/prepare_movi_c_1200.py" ]] || { echo 'Missing prepare_movi_c_1200.py' >&2; exit 1; }
[[ -x "$PREP_PYTHON" ]] || { echo "Python not executable: $PREP_PYTHON" >&2; exit 1; }
[[ "$DATA_ROOT" != /autodl-fs/data/movi_c_flow_v1 ]] || { echo 'Refusing to overwrite the 120/30/30 pilot' >&2; exit 1; }
"$PREP_PYTHON" -c 'import gcsfs, tensorflow, tensorflow_datasets, numpy; print("preparation dependencies: OK")'
mkdir -p "$DATA_ROOT" "$STAGING_ROOT"

echo "Output: $DATA_ROOT"
echo "Temporary single-shard staging: $STAGING_ROOT"
echo 'Target: train=1200, validation=300, test=300'
"$PREP_PYTHON" -u "$SCRIPT_DIR/prepare_movi_c_1200.py" \
  --output-root "$DATA_ROOT" \
  --staging-root "$STAGING_ROOT" \
  --train-count 1200 \
  --validation-count 300 \
  --test-count 300 \
  --seed "$SEED" \
  2>&1 | tee -a "$DATA_ROOT/prepare.log"

for split in train validation test; do
  printf '%s: ' "$split"
  find "$DATA_ROOT/$split" -maxdepth 1 -type f -name '*.npz' | wc -l
done
echo "Data preparation completed: $DATA_ROOT"
