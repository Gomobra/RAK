#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "rak/core/tensor.h"
#include "rak/core/autograd.h"
#include "rak/ops/gemm.h"
#include "rak/ops/elementwise.h"
#include "rak/ops/reduce.h"
#include "rak/ops/norm.h"
#include "rak/ops/embedding.h"
#include "rak/ops/scan.h"
#include "rak/ops/local_attention.h"
#include "rak/ops/sparse_lookup.h"
#include "rak/ops/sparse_update.h"
#include "rak/train/optimizer.h"
namespace py = pybind11;

// ---------- Tensor binding (buffer protocol, zero-copy dengan NumPy) ----------

static py::object tensor_to_numpy(std::shared_ptr<rak::core::Tensor> t) {
  t->validate();
  py::buffer_info info{
      t->ptr,
      sizeof(float),
      py::format_descriptor<float>::format(),
      static_cast<py::ssize_t>(t->ndim()),
      t->shape,
      // stride dalam elemen -> byte
      [&]() {
        std::vector<py::ssize_t> s(t->stride.size());
        for (size_t i = 0; i < t->stride.size(); ++i) s[i] = t->stride[i] * sizeof(float);
        return s;
      }()};
  return py::array(info);  // share memory (no copy)
}

static std::shared_ptr<rak::core::Tensor> numpy_to_tensor(py::buffer b, bool requires_grad) {
  py::buffer_info info = b.request();
  if (info.format != py::format_descriptor<float>::format()) {
    throw std::runtime_error("Tensor: buffer harus bertipe float32");
  }
  if (!info.ndim) {
    throw std::runtime_error("Tensor: buffer harus setidaknya 1 dimensi");
  }
  std::vector<size_t> shape(info.ndim);
  std::vector<size_t> stride(info.ndim);
  for (size_t i = 0; i < info.ndim; ++i) {
    shape[i] = static_cast<size_t>(info.shape[i]);
    stride[i] = static_cast<size_t>(info.strides[i]) / sizeof(float);
  }
  // Buat view tanpa salin ke buffer NumPy. Simpan buffer sebagai pemilik agar
  // pointer eksternal tidak dibuang GC sementara tensor view masih memakainya
  // (menghindari use-after-free).
  auto t = std::make_shared<rak::core::Tensor>(
      static_cast<float*>(info.ptr), shape, stride);
  t->owns_data = false;
  t->requires_grad = requires_grad;
  t->is_leaf = requires_grad;
  if (requires_grad) t->grad.assign(t->numel(), 0.0f);
  // Simpan buffer object sebagai pemilik lifetime (shared agar tensor dan pemanggil
  // dapat membaginya, dan buffer tidak dibuang sebelum tensor selesai dipakai).
  t->py_owner = std::shared_ptr<void>(new py::object(std::move(b)),
                                      [](void* p) { delete static_cast<py::object*>(p); });
  return t;
}

PYBIND11_MODULE(_rak, m) {
  m.doc() = "RAK native module (C++ engine binding).";

  // ---------- DType ----------
  py::enum_<rak::core::DType>(m, "DType")
      .value("FLOAT32", rak::core::DType::kFloat32)
      .export_values();

  // ---------- Tensor ----------
  py::class_<rak::core::Tensor, std::shared_ptr<rak::core::Tensor>>(m, "Tensor")
      .def(py::init([](const std::vector<size_t>& shape, bool requires_grad) {
        return std::make_shared<rak::core::Tensor>(shape, requires_grad);
      }))
      .def(py::init([](const std::vector<size_t>& shape,
                       const std::vector<float>& values,
                       bool requires_grad) {
        return std::make_shared<rak::core::Tensor>(shape, values, requires_grad);
      }))
      .def("shape", [](const std::shared_ptr<rak::core::Tensor>& t) {
        return std::vector<size_t>(t->shape);
      })
      .def("stride", [](const std::shared_ptr<rak::core::Tensor>& t) {
        return std::vector<size_t>(t->stride);
      })
      .def("ndim", &rak::core::Tensor::ndim)
      .def("numel", &rak::core::Tensor::numel)
      .def("at", [](std::shared_ptr<rak::core::Tensor> t, size_t i) -> float& {
        return t->at(i);
      }, py::is_operator())
      .def("at", [](const std::shared_ptr<rak::core::Tensor>& t, size_t i) -> float {
        return t->at(i);
      })
      .def("reshape", &rak::core::Tensor::reshape)
      .def("transpose", &rak::core::Tensor::transpose)
      .def("summary", &rak::core::Tensor::summary)
      .def_property("requires_grad",
                    [](const std::shared_ptr<rak::core::Tensor>& t) { return t->requires_grad; },
                    [](std::shared_ptr<rak::core::Tensor> t, bool v) {
                      t->requires_grad = v;
                      t->is_leaf = v;
                      t->grad.assign(t->numel(), 0.0f);
                    })
      .def_property_readonly("grad", [](std::shared_ptr<rak::core::Tensor> t) {
        return t->grad;
      })
      .def("zero_grad", [](std::shared_ptr<rak::core::Tensor> t) {
        std::fill(t->grad.begin(), t->grad.end(), 0.0f);
      })
      .def("to_numpy", &tensor_to_numpy)
      .def("to_list", [](std::shared_ptr<rak::core::Tensor> t) {
        std::vector<float> v(t->numel());
        for (size_t i = 0; i < t->numel(); ++i) v[i] = t->ptr[i];
        return v;
      })
      .def("validate", &rak::core::Tensor::validate);

  m.def("from_numpy", &numpy_to_tensor,
        py::arg("buffer"), py::arg("requires_grad") = false,
        "Buat Tensor view dari buffer Python (NumPy) tanpa salin data.");

  m.def("zeros", [](const std::vector<size_t>& shape, bool requires_grad) {
    return std::make_shared<rak::core::Tensor>(shape, requires_grad);
  }, "Buat tensor nol.");
  // ---------- Autograd ----------
  m.def("backward", [](std::shared_ptr<rak::core::Tensor> loss, bool retain_graph) {
    rak::core::g_autograd.backward(loss, retain_graph);
  }, py::arg("loss"), py::arg("retain_graph") = false,
     "Jalankan backward (reverse-mode autograd).");

  m.def("autograd_reset", []() { rak::core::g_autograd.reset(); },
        "Reset tape autograd.");

  // ---------- Ops ----------
  m.def("gemm", &rak::ops::gemm, py::arg("A"), py::arg("B"),
        "GEMM: C = A @ B.");
  m.def("add", &rak::ops::add, py::arg("A"), py::arg("B"),
        "Elementwise add dengan broadcast.");
  m.def("mul", &rak::ops::mul, py::arg("A"), py::arg("B"),
        "Elementwise mul dengan broadcast.");
  m.def("silu", &rak::ops::silu, py::arg("X"), "SiLU aktivasi.");
  m.def("gelu", &rak::ops::gelu, py::arg("X"), "GELU aktivasi (aproksimasi tanh).");
  m.def("sum", &rak::ops::sum, py::arg("X"), py::arg("axis") = -1, "Sum along axis.");
  m.def("mean", &rak::ops::mean, py::arg("X"), py::arg("axis") = -1, "Mean along axis.");
  m.def("max", &rak::ops::max, py::arg("X"), py::arg("axis") = -1, "Max along axis.");
  m.def("rms_norm", &rak::ops::rms_norm, py::arg("X"), py::arg("weight"),
        py::arg("eps") = 1e-5f, "RMSNorm.");
  m.def("softmax", &rak::ops::softmax, py::arg("X"), "Softmax stabil.");
  m.def("log_softmax", &rak::ops::log_softmax, py::arg("X"), "Log-softmax stabil.");
  m.def("embedding", &rak::ops::embedding, py::arg("X"), py::arg("weight"),
        "Embedding gather.");
  m.def("recurrent_scan", &rak::ops::recurrent_scan, py::arg("X"), py::arg("decay"),
        py::arg("mode") = "parallel", py::arg("direction") = "forward",
        "Scan rekuren linear bergerbang.");
  m.def("local_attention", &rak::ops::local_attention, py::arg("Q"), py::arg("K"),
        py::arg("V"), py::arg("window"),
        "Attention jendela geser.");
  m.def("sparse_lookup_topk", &rak::ops::sparse_lookup_topk, py::arg("query"),
        py::arg("keys"), py::arg("k"),
        "Lookup top-k kunci produk sparse.");
  m.def("sparse_update", &rak::ops::sparse_update, py::arg("slots"), py::arg("indices"),
        py::arg("updates"), py::arg("mode") = "add",
        "Update sparse (scatter) slot terpilih.");

  // ---------- Sparse lookup result ----------
  py::class_<rak::ops::SparseLookupResult>(m, "SparseLookupResult")
      .def_readonly("values", &rak::ops::SparseLookupResult::values)
      .def_readonly("indices", &rak::ops::SparseLookupResult::indices);
  py::class_<rak::train::OptimizerConfig>(m, "OptimizerConfig")
      .def(py::init<>())
      .def_readwrite("lr", &rak::train::OptimizerConfig::lr)
      .def_readwrite("weight_decay", &rak::train::OptimizerConfig::weight_decay)
      .def_readwrite("beta1", &rak::train::OptimizerConfig::beta1)
      .def_readwrite("beta2", &rak::train::OptimizerConfig::beta2)
      .def_readwrite("eps", &rak::train::OptimizerConfig::eps)
      .def_readwrite("grad_clip", &rak::train::OptimizerConfig::grad_clip)
      .def_readwrite("warmup_steps", &rak::train::OptimizerConfig::warmup_steps)
      .def_readwrite("total_steps", &rak::train::OptimizerConfig::total_steps)
      .def_readwrite("min_lr", &rak::train::OptimizerConfig::min_lr);

  py::class_<rak::train::AdamW, std::shared_ptr<rak::train::AdamW>>(m, "AdamW")
      .def(py::init<const rak::train::OptimizerConfig&>())
      .def("add_parameter", &rak::train::AdamW::add_parameter, py::arg("param"))
      .def("step", &rak::train::AdamW::step, py::arg("global_step"))
      .def("clip_gradients", &rak::train::AdamW::clip_gradients, py::arg("max_norm"))
      .def("global_grad_norm", &rak::train::AdamW::global_grad_norm)
      .def("current_lr", &rak::train::AdamW::current_lr)
      .def("save_state", &rak::train::AdamW::save_state)
      .def("load_state", &rak::train::AdamW::load_state);

  py::class_<rak::train::LRScheduler>(m, "LRScheduler")
      .def(py::init<float, int64_t, int64_t, float>(),
           py::arg("base_lr"), py::arg("warmup_steps"), py::arg("total_steps"),
           py::arg("min_lr") = 0.0f)
      .def("lr_at", &rak::train::LRScheduler::lr_at, py::arg("step"));
}
