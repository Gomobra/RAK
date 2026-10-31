#include "rak/core/tensor.h"
#include "rak/core/allocator.h"

#include <cstring>
#include <numeric>

namespace rak::core {

size_t Tensor::numel() const {
  size_t n = 1;
  for (size_t d : shape) n *= d;
  return n;
}

Tensor::Tensor(std::vector<size_t> shape, bool requires_grad)
    : shape(std::move(shape)), requires_grad(requires_grad), is_leaf(true) {
  size_t n = numel();
  stride.resize(this->shape.size());
  size_t acc = 1;
  for (size_t i = this->shape.size(); i-- > 0;) {
    stride[i] = acc;
    acc *= this->shape[i];
  }
  data.assign(n, 0.0f);
  ptr = data.data();
  grad.assign(n, 0.0f);
}

Tensor::Tensor(std::vector<size_t> shape, std::vector<float> values, bool requires_grad)
    : shape(std::move(shape)),
      data(std::move(values)),
      requires_grad(requires_grad),
      is_leaf(true),
      owns_data(true) {
  size_t n = numel();
  if (data.size() != n) {
    throw std::invalid_argument("Tensor: jumlah data tidak cocok dengan shape");
  }
  stride.resize(this->shape.size());
  size_t acc = 1;
  for (size_t i = this->shape.size(); i-- > 0;) {
    stride[i] = acc;
    acc *= this->shape[i];
  }
  ptr = data.data();
  grad.assign(n, 0.0f);
}

Tensor::Tensor(float* ptr, std::vector<size_t> shape, std::vector<size_t> stride)
    : shape(std::move(shape)), stride(std::move(stride)), is_leaf(false), ptr(ptr), owns_data(false) {
  size_t n = numel();
  grad.assign(n, 0.0f);
}

float& Tensor::at(size_t i) {
  if (i >= numel()) throw std::out_of_range("Tensor::at indeks di luar jangkauan");
  return ptr[i];
}

float Tensor::at(size_t i) const {
  if (i >= numel()) throw std::out_of_range("Tensor::at indeks di luar jangkauan");
  return ptr[i];
}

float& Tensor::at(const std::vector<size_t>& idx) {
  return ptr[index(idx)];
}

float Tensor::at(const std::vector<size_t>& idx) const {
  return ptr[index(idx)];
}

size_t Tensor::index(const std::vector<size_t>& idx) const {
  if (idx.size() != shape.size()) {
    throw std::invalid_argument("Tensor::index jumlah indeks tidak cocok dengan ndim");
  }
  size_t off = 0;
  for (size_t i = 0; i < idx.size(); ++i) {
    if (idx[i] >= shape[i]) throw std::out_of_range("Tensor::at indeks di luar jangkauan");
    off += idx[i] * stride[i];
  }
  return off;
}

Tensor Tensor::reshape(std::vector<size_t> new_shape) const {
  size_t n = 1;
  for (size_t d : new_shape) n *= d;
  if (n != numel()) {
    throw std::invalid_argument("Tensor::reshape total elemen tidak cocok");
  }
  Tensor t;
  t.dtype = dtype;
  t.shape = std::move(new_shape);
  t.stride.resize(t.shape.size());
  size_t acc = 1;
  for (size_t i = t.shape.size(); i-- > 0;) {
    t.stride[i] = acc;
    acc *= t.shape[i];
  }
  t.ptr = ptr;
  t.owns_data = false;
  t.version = version;
  return t;
}

Tensor Tensor::transpose(size_t a, size_t b) const {
  if (a >= shape.size() || b >= shape.size()) {
    throw std::out_of_range("Tensor::transpose axis di luar jangkauan");
  }
  Tensor t;
  t.dtype = dtype;
  t.shape = shape;
  t.stride = stride;
  std::swap(t.shape[a], t.shape[b]);
  std::swap(t.stride[a], t.stride[b]);
  t.ptr = ptr;
  t.owns_data = false;
  t.version = version;
  return t;
}

std::string Tensor::summary() const {
  std::ostringstream os;
  os << "Tensor(shape=[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i) os << ", ";
    os << shape[i];
  }
  os << "], numel=" << numel() << ", dtype=";
  os << (dtype == DType::kFloat32 ? "float32" : "?");
  os << ", requires_grad=" << (requires_grad ? "true" : "false");
  os << ", is_leaf=" << (is_leaf ? "true" : "false");
  os << ")";
  return os.str();
}

void Tensor::validate() const {
  fprintf(stderr, "[DBG validate] shape.size=%zu stride.size=%zu numel=%zu\n",
          shape.size(), stride.size(), numel());
  if (stride.size() != shape.size()) {
    throw std::runtime_error("Tensor: panjang stride tidak cocok dengan shape");
  }
  if (ptr == nullptr && numel() > 0) {
    throw std::runtime_error("Tensor: ptr null tetapi numel > 0");
  }
}

}  // namespace rak::core
