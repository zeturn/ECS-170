#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data/stage_2_data"

TRAIN_FILE="$DATA_DIR/train.csv"
TEST_FILE="$DATA_DIR/test.csv"

if [[ ! -f "$TRAIN_FILE" || ! -f "$TEST_FILE" ]]; then
  echo "Missing Stage 2 dataset files."
  echo "Expected files:"
  echo "  $TRAIN_FILE"
  echo "  $TEST_FILE"
  echo "Please place your train/test files first, then rerun this script."
  exit 1
fi

cd "$SCRIPT_DIR"
python script_mlp.py
