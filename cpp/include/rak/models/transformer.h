#ifndef RAK_MODELS_TRANSFORMER_H
#define RAK_MODELS_TRANSFORMER_H

#include "rak/models/model.h"
#include "rak/ops/gemm.h"
#include "rak/ops/elementwise.h"
#include "rak/ops/reduce.h"
#include "rak/ops/norm.h"

namespace rak::models {

// Transformer Decoder Murni (F1.2.1).
// Arsitektur: token embedding (+ positional learned) -> N blok decoder
//   (causal MHA -> residual -> RMSNorm -> FFN -> residual -> RMSNorm)
//   -> output head (tied dengan embedding) -> logits.
// Causal masking diterapkan via matriks mask + ops `add` sebelum softmax
// (karena local_attention tidak punya causal mask bawaan).
class Transformer : public BaseModel {
 public:
  struct Config {
    uint64_t d_model = 256;
    uint64_t n_layer = 8;
    uint64_t d_ff = 1024;
    uint64_t n_head = 8;
    uint64_t vocab_size = 16384;
    uint64_t max_seq = 1024;
    float eps = 1e-5f;
    float lr = 1e-3f;
    float weight_decay = 0.01f;
    float grad_clip = 1.0f;
    int64_t warmup_steps = 500;
    int64_t total_steps = 10000;
    float min_lr = 1e-5f;
  };

  Transformer(const Config& cfg);

  std::shared_ptr<core::Tensor> forward(
      const std::shared_ptr<core::Tensor>& input_ids) override;

  std::shared_ptr<core::Tensor> compute_loss(
      const std::shared_ptr<core::Tensor>& logits,
      const std::shared_ptr<core::Tensor>& target_ids) override;

  uint64_t num_parameters() const override { return num_params_; }
  std::string name() const override { return "Transformer"; }
  std::vector<std::shared_ptr<core::Tensor>> parameters() override { return params_; }

  const Config& config() const { return cfg_; }

  // Latih satu batch: forward + loss + backward. Kembalikan loss nilai.
  float train_step(const std::shared_ptr<core::Tensor>& input_ids,
                   const std::shared_ptr<core::Tensor>& target_ids,
                   float current_lr);

 private:
  Config cfg_;
  uint64_t num_params_ = 0;
  std::vector<std::shared_ptr<core::Tensor>> params_;

  // Parameter per blok dan global.
  struct BlockParams {
    std::shared_ptr<core::Tensor> w_q, w_k, w_v, w_o;  // MHA
    std::shared_ptr<core::Tensor> w1, w2;              // FFN
    std::shared_ptr<core::Tensor> w_r1, w_r2;          // RMSNorm weight
  };
  std::shared_ptr<core::Tensor> w_emb_;       // token embedding (tied)
  std::shared_ptr<core::Tensor> w_pos_;       // positional embedding
  std::vector<BlockParams> blocks_;
};

}  // namespace rak::models

#endif  // RAK_MODELS_TRANSFORMER_H
