#ifndef RAK_CORE_TENSOR_H
#define RAK_CORE_TENSOR_H

#include <cstddef>
#include <cstdint>
#include <vector>
#include <string>
#include <memory>
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <algorithm>

#include "rak/core/allocator.h"

namespace rak::core {

class GradNode;

// Tipe data yang didukung tensor.
enum class DType : uint8_t {
  kFloat32 = 0,
};

inline size_t dtype_sizeof(DType dt) {
  switch (dt) {
    case DType::kFloat32: return sizeof(float);
  }
  return 0;
}

// Tensor n-dimensi float32. Data disimpan dalam satu buffer flat (row-major).
// Mendukung view tanpa salin (reshape, transpose via stride).
struct Tensor {
  DType dtype = DType::kFloat32;
  std::vector<size_t> shape;      // dimensi per axis
  std::vector<size_t> stride;     // elemen (bukan byte) antara elemen berurutan per axis
  std::vector<float> data;        // pemilik buffer (jika owns_data == true)
  float* ptr = nullptr;           // pointer ke data (bisa menunjuk ke data milik orang lain untuk view)
  bool owns_data = true;          // apakah tensor ini yang mengelola `data`
  bool requires_grad = false;     // apakah gradien perlu ditracking untuk tensor ini
  std::vector<float> grad;        // buffer gradien (hanya untuk leaf yang requires_grad)
  bool is_leaf = false;           // apakah tensor ini adalah leaf (parameter/input)
  uint64_t version = 0;           // untuk deteksi in-place mutation pada saved tensor
  std::shared_ptr<GradNode> grad_fn;  // node operasi yang menghasilkan tensor ini (null untuk leaf)
  // Pemilik buffer eksternal (misalnya buffer NumPy) agar pointer eksternal tidak
  // dibuang GC sementara tensor view masih memakainya. Kosong jika owns_data == true.
  std::shared_ptr<void> py_owner;

  Tensor() = default;

  // Buat tensor dengan shape tertentu, dialokasi nol.
  explicit Tensor(std::vector<size_t> shape, bool requires_grad = false);

  // Buat tensor dari data flat (copy).
  Tensor(std::vector<size_t> shape, std::vector<float> values, bool requires_grad = false);

  // Buat view tanpa salin: ptr menunjuk ke buffer eksternal, owns_data = false.
  Tensor(float* ptr, std::vector<size_t> shape, std::vector<size_t> stride);

  size_t ndim() const { return shape.size(); }
  size_t numel() const;

  // Akses elemen linear (untuk debug / kecil).
  float& at(size_t i);
  float at(size_t i) const;

  // Akses elemen dengan indeks multi-dimensi.
  float& at(const std::vector<size_t>& idx);
  float at(const std::vector<size_t>& idx) const;

  // Indeks linear dari indeks multi-dimensi.
  size_t index(const std::vector<size_t>& idx) const;

  // Reshape view (tidak menyalin data), total elemen harus sama.
  Tensor reshape(std::vector<size_t> new_shape) const;

  // Transpose dua axis.
  Tensor transpose(size_t a, size_t b) const;

  // Cetak ringkas (untuk debugging).
  std::string summary() const;

  // Validasi konsistensi shape/stride/ptr.
  void validate() const;
};

}  // namespace rak::core

#endif  // RAK_CORE_TENSOR_H
