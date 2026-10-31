#!/usr/bin/env bash
# run_tests.sh — jalankan test C++ dan Python sekaligus.
# Pemakaian:  bash scripts/run_tests.sh [build_dir]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${1:-$ROOT/build}"

echo "=== Menjalankan test Python ==="
if [ -d "$BUILD_DIR/cpp/python" ]; then
  export PYTHONPATH="$BUILD_DIR/cpp/python:$PYTHONPATH"
fi
python3 -m pytest "$ROOT/tests/" -v --timeout=60

echo "=== Selesai semua test ==="
