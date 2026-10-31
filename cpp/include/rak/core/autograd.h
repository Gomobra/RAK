#ifndef RAK_CORE_AUTOGRAD_H
#define RAK_CORE_AUTOGRAD_H

#include "rak/core/tensor.h"

#include <functional>
#include <vector>
#include <memory>
#include <unordered_map>
#include <stack>
#include <stdexcept>

namespace rak::core {

// Simpul dalam grafik komputasi (tape). Setiap operasi non-leaf menghasilkan
// satu GradNode yang menyimpan fungsi backward dan referensi ke inputnya.
class GradNode {
 public:
  using BackwardFn = std::function<std::vector<std::vector<float>>(const std::vector<float>& grad_output)>;

  GradNode(BackwardFn backward, std::vector<std::weak_ptr<Tensor>> inputs)
      : backward_(std::move(backward)), inputs_(std::move(inputs)) {}

  // Jalankan backward dengan grad_output, kembalikan gradien per input.
  std::vector<std::vector<float>> apply(const std::vector<float>& grad_output) {
    return backward_(grad_output);
  }

  const std::vector<std::weak_ptr<Tensor>>& inputs() const { return inputs_; }

 private:
  BackwardFn backward_;
  std::vector<std::weak_ptr<Tensor>> inputs_;
};

// Tape (graf komputasi) reverse-mode. Terekam setiap operasi forward dan
// menyediakan backward() yang menurunkan gradien ke semua leaf.
class Autograd {
 public:
  Autograd() = default;

  // Catat node operasi baru ke tape.
  void record(std::shared_ptr<GradNode> node) {
    nodes_.push_back(std::move(node));
  }

  // Jalankan backward dari tensor loss (skalar). Gradien di-akumulasi ke
  // buffer grad setiap leaf yang requires_grad. Grafik dibebaskan setelah
  // backward kecuali retain_graph = true.
  void backward(std::shared_ptr<Tensor> loss, bool retain_graph = false);

  // Hapus semua catatan tape (grafik dibebaskan).
  void reset();

  size_t node_count() const { return nodes_.size(); }

 private:
  std::vector<std::shared_ptr<GradNode>> nodes_;
};

// Instance tape global (default). Mode eager: satu tape aktif sepanjang sesi.
extern Autograd g_autograd;

// Buat tensor hasil operasi dan catat node gradien-nya. Fungsi bantu untuk op.
std::shared_ptr<Tensor> make_op_output(
    std::vector<std::shared_ptr<Tensor>> inputs,
    std::vector<float> forward_values,
    std::vector<size_t> shape,
    GradNode::BackwardFn backward);

}  // namespace rak::core

#endif  // RAK_CORE_AUTOGRAD_H
