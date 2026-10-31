#include "rak/ops/reduce.h"

#include <stdexcept>
#include <cmath>
#include <limits>

namespace rak::ops {

// axis == -1 berarti "reduce semua elemen ke skalar" (output shape {1}).
// axis >= 0 berarti reduce satu axis tertentu.

static std::vector<size_t> reduce_shape_axis(
    const std::vector<size_t>& shape, int64_t axis) {
  std::vector<size_t> out;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (static_cast<int64_t>(i) != axis) out.push_back(shape[i]);
  }
  if (out.empty()) out.push_back(1);
  return out;
}

static std::vector<size_t> full_coord(
    const std::vector<size_t>& shape, const std::vector<size_t>& coord,
    int64_t ax, size_t k) {
  std::vector<size_t> fc(shape.size());
  size_t oi = 0;
  for (size_t d = 0; d < shape.size(); ++d) {
    if (static_cast<int64_t>(d) == ax) {
      fc[d] = k;
    } else {
      fc[d] = coord[oi];
      ++oi;
    }
  }
  return fc;
}

static size_t linear_index(const std::vector<size_t>& coord,
                           const std::vector<size_t>& stride) {
  size_t idx = 0;
  for (size_t d = 0; d < coord.size(); ++d) idx += coord[d] * stride[d];
  return idx;
}

// ---------- sum ----------
std::shared_ptr<core::Tensor> sum(
    const std::shared_ptr<core::Tensor>& X, int64_t axis) {
  if (!X) throw std::invalid_argument("sum: input null");

  std::vector<size_t> out_shape;
  size_t reduce_size = 1;
  bool reduce_all = (axis == -1);
  int64_t ax = reduce_all ? -1 : axis;

  if (reduce_all) {
    out_shape = {1};
    reduce_size = X->numel();
  } else {
    if (ax < 0) ax = static_cast<int64_t>(X->ndim()) + ax;
    if (ax < 0 || ax >= static_cast<int64_t>(X->ndim())) {
      throw std::invalid_argument("sum: axis di luar jangkauan");
    }
    out_shape = reduce_shape_axis(X->shape, ax);
    reduce_size = X->shape[ax];
  }

  size_t out_numel = 1;
  for (size_t d : out_shape) out_numel *= d;

  std::vector<float> out(out_numel, 0.0f);

  if (reduce_all) {
    float s = 0.0f;
    for (size_t i = 0; i < X->numel(); ++i) s += X->ptr[i];
    out[0] = s;
  } else {
    for (size_t i = 0; i < out_numel; ++i) {
      std::vector<size_t> coord(out_shape.size());
      size_t r = i;
      for (size_t d = out_shape.size(); d-- > 0;) {
        coord[d] = r % out_shape[d];
        r /= out_shape[d];
      }
      float s = 0.0f;
      for (size_t k = 0; k < reduce_size; ++k) {
        std::vector<size_t> fc = full_coord(X->shape, coord, ax, k);
        s += X->ptr[linear_index(fc, X->stride)];
      }
      out[i] = s;
    }
  }

  return core::make_op_output(
      {X},
      std::move(out),
      out_shape,
      [X, axis, reduce_size, reduce_all, ax](
          const std::vector<float>& grad_out) {
        size_t xnumel = X->numel();
        std::vector<float> g(xnumel, 0.0f);

        if (reduce_all) {
          float go = grad_out[0];
          for (size_t i = 0; i < xnumel; ++i) g[i] += go;
        } else {
          int64_t ax_local = ax;
          std::vector<size_t> out_shape = reduce_shape_axis(X->shape, ax_local);
          size_t out_numel = grad_out.size();
          for (size_t i = 0; i < out_numel; ++i) {
            std::vector<size_t> coord(out_shape.size());
            size_t r = i;
            for (size_t d = out_shape.size(); d-- > 0;) {
              coord[d] = r % out_shape[d];
              r /= out_shape[d];
            }
            for (size_t k = 0; k < reduce_size; ++k) {
              std::vector<size_t> fc = full_coord(X->shape, coord, ax_local, k);
              size_t idx = linear_index(fc, X->stride);
              g[idx] += grad_out[i];
            }
          }
        }
        return std::vector<std::vector<float>>{g};
      });
}

// ---------- mean ----------
std::shared_ptr<core::Tensor> mean(
    const std::shared_ptr<core::Tensor>& X, int64_t axis) {
  if (!X) throw std::invalid_argument("mean: input null");

  bool reduce_all = (axis == -1);
  size_t reduce_size;
  int64_t ax = reduce_all ? -1 : axis;

  if (reduce_all) {
    reduce_size = X->numel();
  } else {
    if (ax < 0) ax = static_cast<int64_t>(X->ndim()) + ax;
    if (ax < 0 || ax >= static_cast<int64_t>(X->ndim())) {
      throw std::invalid_argument("mean: axis di luar jangkauan");
    }
    reduce_size = X->shape[ax];
  }

  auto s = sum(X, axis);
  size_t n = s->numel();
  std::vector<float> out(n);
  for (size_t i = 0; i < n; ++i) out[i] = s->ptr[i] / static_cast<float>(reduce_size);

  return core::make_op_output(
      {X},
      std::move(out),
      s->shape,
      [X, axis, reduce_size, reduce_all, ax](const std::vector<float>& grad_out) {
        size_t xnumel = X->numel();
        std::vector<float> g(xnumel, 0.0f);

        if (reduce_all) {
          float v = grad_out[0] / static_cast<float>(reduce_size);
          for (size_t i = 0; i < xnumel; ++i) g[i] += v;
        } else {
          int64_t ax_local = ax;
          std::vector<size_t> out_shape = reduce_shape_axis(X->shape, ax_local);
          size_t out_numel = grad_out.size();
          for (size_t i = 0; i < out_numel; ++i) {
            std::vector<size_t> coord(out_shape.size());
            size_t r = i;
            for (size_t d = out_shape.size(); d-- > 0;) {
              coord[d] = r % out_shape[d];
              r /= out_shape[d];
            }
            for (size_t k = 0; k < reduce_size; ++k) {
              std::vector<size_t> fc = full_coord(X->shape, coord, ax_local, k);
              size_t idx = linear_index(fc, X->stride);
              g[idx] += grad_out[i] / static_cast<float>(reduce_size);
            }
          }
        }
        return std::vector<std::vector<float>>{g};
      });
}

// ---------- max ----------
std::shared_ptr<core::Tensor> max(
    const std::shared_ptr<core::Tensor>& X, int64_t axis) {
  if (!X) throw std::invalid_argument("max: input null");

  std::vector<size_t> out_shape;
  size_t reduce_size = 1;
  bool reduce_all = (axis == -1);
  int64_t ax = reduce_all ? -1 : axis;

  if (reduce_all) {
    out_shape = {1};
    reduce_size = X->numel();
  } else {
    if (ax < 0) ax = static_cast<int64_t>(X->ndim()) + ax;
    if (ax < 0 || ax >= static_cast<int64_t>(X->ndim())) {
      throw std::invalid_argument("max: axis di luar jangkauan");
    }
    out_shape = reduce_shape_axis(X->shape, ax);
    reduce_size = X->shape[ax];
  }

  size_t out_numel = 1;
  for (size_t d : out_shape) out_numel *= d;

  std::vector<float> out(out_numel);
  std::vector<size_t> argmax(out_numel, 0);

  if (reduce_all) {
    float best = -std::numeric_limits<float>::infinity();
    size_t best_k = 0;
    for (size_t k = 0; k < reduce_size; ++k) {
      float v = X->ptr[k];
      if (v > best) { best = v; best_k = k; }
    }
    out[0] = best;
    argmax[0] = best_k;
  } else {
    for (size_t i = 0; i < out_numel; ++i) {
      std::vector<size_t> coord(out_shape.size());
      size_t r = i;
      for (size_t d = out_shape.size(); d-- > 0;) {
        coord[d] = r % out_shape[d];
        r /= out_shape[d];
      }
      float best = -std::numeric_limits<float>::infinity();
      size_t best_k = 0;
      for (size_t k = 0; k < reduce_size; ++k) {
        std::vector<size_t> fc = full_coord(X->shape, coord, ax, k);
        size_t idx = linear_index(fc, X->stride);
        float v = X->ptr[idx];
        if (v > best) { best = v; best_k = k; }
      }
      out[i] = best;
      argmax[i] = best_k;
    }
  }

  return core::make_op_output(
      {X},
      std::move(out),
      out_shape,
      [X, axis, reduce_size, reduce_all, ax, argmax = std::move(argmax)](
          const std::vector<float>& grad_out) {
        size_t xnumel = X->numel();
        std::vector<float> g(xnumel, 0.0f);

        if (reduce_all) {
          size_t k = argmax[0];
          g[k] += grad_out[0];
        } else {
          int64_t ax_local = ax;
          std::vector<size_t> out_shape = reduce_shape_axis(X->shape, ax_local);
          size_t out_numel = grad_out.size();
          for (size_t i = 0; i < out_numel; ++i) {
            std::vector<size_t> coord(out_shape.size());
            size_t r = i;
            for (size_t d = out_shape.size(); d-- > 0;) {
              coord[d] = r % out_shape[d];
              r /= out_shape[d];
            }
            size_t k = argmax[i];
            std::vector<size_t> fc = full_coord(X->shape, coord, ax_local, k);
            size_t idx = linear_index(fc, X->stride);
            g[idx] += grad_out[i];
          }
        }
        return std::vector<std::vector<float>>{g};
      });
}

}  // namespace rak::ops
