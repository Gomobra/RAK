#!/usr/bin/env bash
# run_tests.sh — jalankan test C++ dan Python sekaligus.
# Pemakaian:  bash scripts/run_tests.sh [--mode termux|ci|kaggle] [build_dir]
#
# Mode:
#   termux (default) — build tanpa test C++ (Catch2 sulit di Termux),
#                      jalankan pytest smoke test saja.
#   ci               — GitHub Actions CPU: build dengan RAK_BUILD_TESTS=ON,
#                      jalankan rak_tests + pytest.
#   kaggle           — Kaggle GPU: build dengan RAK_CUDA=ON,
#                      jalankan rak_tests + rak_bench + pytest.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="termux"
BUILD_DIR="$ROOT/build"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"; shift 2 ;;
    --build-dir)
      BUILD_DIR="$2"; shift 2 ;;
    *)
      echo "Argumen tidak dikenal: $1" >&2; exit 1 ;;
  esac
done

export PYTHONPATH="$BUILD_DIR/cpp/python:$PYTHONPATH"

echo "=== Mode: $MODE ==="
echo "=== Build directory: $BUILD_DIR ==="

case "$MODE" in
  termux)
    echo "Menjalankan test Python (smoke) untuk Termux..."
    if [ -d "$BUILD_DIR/cpp/python" ]; then
      python3 -m pytest "$ROOT/tests/smoke/" -v --timeout=60
    else
      echo "Peringatan: build/cpp/python tidak ada, jalankan build_termux.sh dulu" >&2
      exit 1
    fi
    ;;
  ci)
    echo "Menjalankan test C++ (Catch2) dan Python untuk CI..."
    if [ ! -x "$BUILD_DIR/cpp/rak_tests" ]; then
      echo "Peringatan: rak_tests tidak ada, build dulu dengan RAK_BUILD_TESTS=ON" >&2
      exit 1
    fi
    "$BUILD_DIR/cpp/rak_tests"
    echo "Menjalankan test Python (smoke + parity)..."
    python3 -m pytest "$ROOT/tests/" -v --timeout=60
    ;;
  kaggle)
    echo "Menjalankan test C++, benchmark, dan Python untuk Kaggle..."
    if [ ! -x "$BUILD_DIR/cpp/rak_tests" ]; then
      echo "Peringatan: rak_tests tidak ada, build dulu dengan build_kaggle.sh" >&2
      exit 1
    fi
    "$BUILD_DIR/cpp/rak_tests"
    if [ -x "$BUILD_DIR/cpp/rak_bench" ]; then
      mkdir -p "$ROOT/experiments/reports"
      "$BUILD_DIR/cpp/rak_bench" | tee "$ROOT/experiments/reports/bench_report_kaggle.txt"
    else
      echo "Peringatan: rak_bench tidak ada, lewati benchmark" >&2
    fi
    echo "Menjalankan test Python (smoke + parity)..."
    python3 -m pytest "$ROOT/tests/" -v --timeout=60
    ;;
  *)
    echo "Mode tidak dikenal: $MODE (pilih: termux, ci, kaggle)" >&2
    exit 1 ;;
esac

echo "=== Selesai semua test ($MODE) ==="
