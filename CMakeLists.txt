# RAK (Recurrent Adaptive Knowledge) — Vbeta
# Build dengan CMake. Satu basis kode untuk Termux (clang), GitHub Actions (CPU), dan Kaggle (CUDA opsional).
# CUDA diaktifkan hanya lewat -DRAK_CUDA=ON (default OFF).

cmake_minimum_required(VERSION 3.18)
project(rak LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Opsi CUDA (opsional, default OFF). Semua harus jalan di CPU.
option(RAK_CUDA "Aktifkan backend CUDA untuk op panas" OFF)
option(RAK_BUILD_TESTS "Bangun test C++ (Catch2)" ON)
option(RAK_BUILD_BENCH "Bangun benchmark" ON)
option(RAK_BUILD_PYTHON "Bangun binding Python (_rak)" ON)
option(RAK_CPU_DISPATCH "Aktifkan deteksi fitur CPU (NEON/AVX2) dengan fallback skalar" ON)

# Peringatan yang ketat untuk deteksi bug awal di semua toolchain.
add_compile_options(-Wall -Wextra -Wno-unused-parameter)

# Subdirektori inti.
add_subdirectory(cpp)

# Ekspor variabel agar skrip Python/paket tahu di mana hasil build berada.
set(RAK_SOURCE_DIR ${PROJECT_SOURCE_DIR})
set(RAK_BINARY_DIR ${PROJECT_BINARY_DIR})
set(RAK_CUDA ${RAK_CUDA})
