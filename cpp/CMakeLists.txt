# cpp/CMakeLists.txt — target inti RAK (tensor, autograd, op, train), test, benchmark, dan binding Python.

include(CheckCXXCompilerFlag)
include(FetchContent)

# Catch2 (header-only, ringan di Termux). Digunakan untuk test C++.
if(RAK_BUILD_TESTS)
  FetchContent_Declare(
    catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.4.0
  )
  # Di lingkungan tanpa internet, biarkan FetchContent gagal dengan pesan jelas.
  set(FETCHCONTENT_FULLY_DISCONNECTED OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(catch2)
endif()

# pybind11 (untuk binding Python). Coba sistem dulu.
if(RAK_BUILD_PYTHON)
  find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
  find_package(pybind11 QUIET)
  if(NOT pybind11_FOUND)
    # Coba cari include pybind11 dari instalasi pip.
    execute_process(
      COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_include())"
      OUTPUT_VARIABLE PYBIND11_INCLUDE_DIR
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET)
    if(PYBIND11_INCLUDE_DIR)
      add_library(pybind11::module INTERFACE IMPORTED)
      target_include_directories(pybind11::module INTERFACE ${PYBIND11_INCLUDE_DIR})
      set(pybind11_FOUND TRUE)
    endif()
  endif()
  if(NOT pybind11_FOUND)
    message(FATAL_ERROR "pybind11 tidak ditemukan. Pasang dengan: python3 -m pip install pybind11")
  endif()
endif()

# ---------- Deteksi fitur CPU (NEON/AVX2) dengan fallback skalar ----------
set(RAK_HAVE_NEON OFF)
set(RAK_HAVE_AVX2 OFF)
if(RAK_CPU_DISPATCH)
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-march=native" RAK_HAVE_MARCH_NATIVE)
  if(RAK_HAVE_MARCH_NATIVE)
    # Di Termux aarch64 ini akan menghasilkan NEON; di x86-64 menghasilkan AVX2/AVX-512.
    set(RAK_CPU_FLAGS "-march=native")
  else()
    # Fallback eksplisit per arsitektur.
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
      set(RAK_HAVE_NEON ON)
      set(RAK_CPU_FLAGS "-march=armv8-a")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i386")
      set(RAK_HAVE_AVX2 ON)
      set(RAK_CPU_FLAGS "-mavx2 -mfma")
    endif()
  endif()
endif()

# ---------- CUDA (opsional) ----------
set(RAK_CUDA_AVAILABLE OFF)
if(RAK_CUDA)
  find_package(CUDAToolkit REQUIRED)
  set(RAK_CUDA_AVAILABLE ON)
  add_compile_definitions(RAK_CUDA_ENABLED)
endif()

# ---------- Direktori include publik ----------
add_library(rak_core INTERFACE)
target_include_directories(rak_core INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_definitions(rak_core INTERFACE
  RAK_CUDA_AVAILABLE=${RAK_CUDA_AVAILABLE}
)
if(RAK_CPU_FLAGS)
  target_compile_options(rak_core INTERFACE ${RAK_CPU_FLAGS})
endif()
if(RAK_HAVE_NEON)
  target_compile_definitions(rak_core INTERFACE RAK_HAVE_NEON=1)
endif()
if(RAK_HAVE_AVX2)
  target_compile_definitions(rak_core INTERFACE RAK_HAVE_AVX2=1)
endif()

# ---------- Sumber inti (CPU) ----------
set(RAK_CORE_SOURCES
  src/core/tensor.cpp
  src/core/autograd.cpp
  src/core/allocator.cpp
  src/ops/gemm.cpp
  src/ops/elementwise.cpp
  src/ops/reduce.cpp
  src/ops/norm.cpp
  src/ops/embedding.cpp
  src/ops/scan.cpp
  src/ops/local_attention.cpp
  src/ops/sparse_lookup.cpp
  src/ops/sparse_update.cpp
  src/train/optimizer.cpp
  src/train/checkpoint.cpp
  src/testing/grad_check.cpp
)

# Sumber CUDA (hanya jika RAK_CUDA=ON).
set(RAK_CUDA_SOURCES
  src/cuda/gemm.cu
  src/cuda/scan.cu
  src/cuda/sparse_lookup.cu
)

add_library(rak_impl STATIC ${RAK_CORE_SOURCES})
target_link_libraries(rak_impl PUBLIC rak_core)
set_target_properties(rak_impl PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(RAK_CUDA_AVAILABLE)
  target_sources(rak_impl PRIVATE ${RAK_CUDA_SOURCES})
  target_link_libraries(rak_impl PUBLIC CUDA::cudart CUDA::cublas)
endif()

# ---------- Test C++ ----------
if(RAK_BUILD_TESTS)
  add_executable(rak_tests
    tests/test_core.cpp
  )
  target_link_libraries(rak_tests PRIVATE rak_impl Catch2::Catch2WithMain)
  target_include_directories(rak_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  if(RAK_CUDA_AVAILABLE)
    target_compile_definitions(rak_tests PRIVATE RAK_CUDA_ENABLED)
  endif()
endif()

# ---------- Benchmark ----------
if(RAK_BUILD_BENCH)
  add_executable(rak_bench
    apps/bench_gemm.cpp
    apps/bench_ops.cpp
  )
  target_link_libraries(rak_bench PRIVATE rak_impl)
  target_include_directories(rak_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
endif()

# ---------- Binding Python (_rak) ----------
if(RAK_BUILD_PYTHON)
  add_library(_rak MODULE bindings/module.cpp)
  target_link_libraries(_rak PRIVATE rak_impl pybind11::module Python3::Python)
  target_include_directories(_rak PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  # Modul bernama _rak agar dapat diimpor dari Python sebagai src.rak._native.
  set_target_properties(_rak PROPERTIES
    PREFIX ""
    SUFFIX ".so"
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/python
  )
endif()
