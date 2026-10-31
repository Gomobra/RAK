#!/usr/bin/env bash
# run_kaggle_exp.sh — jalankan eksperimen Fase 1 di Kaggle (GPU).
# Pemakaian:  bash scripts/experiments/run_kaggle_exp.sh [exp_name]
#
# Skrip ini mengaktifkan environment, menjalankan eksperimen baseline kecil,
# dan menyimpan checkpoint + log ke experiments/runs/.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

EXP_NAME="${1:-baseline_exp_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="$ROOT/experiments/runs/$EXP_NAME"
REPORT_DIR="$ROOT/experiments/reports"

mkdir -p "$RUN_DIR" "$REPORT_DIR"

echo "=== Eksperimen Kaggle: $EXP_NAME ==="
echo "Run directory: $RUN_DIR"

# Simpan metadata environment
{
  echo "exp_name=$EXP_NAME"
  echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "cuda=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
  echo "python=$(python3 --version 2>&1 || echo 'N/A')"
} > "$RUN_DIR/metadata.txt"

# Build dulu jika belum ada
if [ ! -x "build/cpp/rak_tests" ]; then
  echo "Build RAK untuk Kaggle..."
  bash "$ROOT/scripts/build/build_kaggle.sh" ON
fi

# Jalankan test C++
echo "Menjalankan rak_tests..."
build/cpp/rak_tests | tee "$RUN_DIR/test_cpp.log" || true

# Jalankan benchmark
if [ -x "build/cpp/rak_bench" ]; then
  echo "Menjalankan rak_bench..."
  build/cpp/rak_bench | tee "$RUN_DIR/bench.log" || true
  cp "$RUN_DIR/bench.log" "$REPORT_DIR/bench_report_kaggle.txt" || true
fi

# Jalankan test Python
echo "Menjalankan test Python..."
export PYTHONPATH="build/cpp/python:$PYTHONPATH"
python3 -m pytest "$ROOT/tests/" -v --timeout=60 | tee "$RUN_DIR/test_python.log" || true

echo "=== Eksperimen selesai: $EXP_NAME ==="
echo "Hasil di: $RUN_DIR"
echo "Laporan di: $REPORT_DIR"
