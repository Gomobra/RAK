#ifndef RAK_MODELS_MODEL_H
#define RAK_MODELS_MODEL_H

#include "rak/core/tensor.h"
#include "rak/core/autograd.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <memory>

namespace rak::models {

// Antarmuka dasar model bahasa. Semua model baseline Fase 1 mewarisi ini.
class BaseModel {
 public:
  virtual ~BaseModel() = default;

  // Forward: input token IDs [batch, seq] -> logits [batch, seq, vocab]
  virtual std::shared_ptr<core::Tensor> forward(
      const std::shared_ptr<core::Tensor>& input_ids) = 0;

  // Cross-entropy loss pada logits [batch, seq, vocab] dan target [batch, seq]
  virtual std::shared_ptr<core::Tensor> compute_loss(
      const std::shared_ptr<core::Tensor>& logits,
      const std::shared_ptr<core::Tensor>& target_ids) = 0;

  // Backward dari loss tensor.
  void backward(const std::shared_ptr<core::Tensor>& loss) {
    rak::core::g_autograd.backward(loss);
  }

  // Reset tape autograd.
  void reset_tape() { rak::core::g_autograd.reset(); }

  // Jumlah total parameter.
  virtual uint64_t num_parameters() const = 0;

  // Nama model (untuk logging).
  virtual std::string name() const = 0;

  // Dapatkan semua parameter (leaf tensor yang requires_grad).
  virtual std::vector<std::shared_ptr<core::Tensor>> parameters() = 0;
};

}  // namespace rak::models

#endif  // RAK_MODELS_MODEL_H
