#ifndef RAK_MODELS_HYBRID_H
#define RAK_MODELS_HYBRID_H

#include "rak/models/model.h"
#include "rak/ops/gemm.h"
#include "rak/ops/elementwise.h"
#include "rak/ops/reduce.h"
#include "rak/ops/norm.h"
#include "rak/ops/scan.h"
#include "rak/ops/local_attention.h"
#include "rak/ops/sparse_lookup.h"

namespace rak::models {

// Hybrid Rekuren + Attention Lokal (F1.2.2).
// Setiap blok: recurrent_scan (state kompres) + local_attention (jendela dekat),
// digabung residual, diikuti FFN + RMSNorm. sparse_lookup_topk sebagai memory
// slot bank opsional.
// Tidak menggunakan komponen RAK lain (MoE, memory layer, adaptive loop, dll)
class Hybrid : public BaseModel {
 public:
  struct Config {
    uint64_t d_model = 256;
    uint64_t n_layer = 8;
    uint64_t d_ff = 1024;
    uint64_t vocab_size = 16384;
    uint64_t max_seq = 1024;
    uint64_t window = 64;
    uint64_t n_slot = 512;
    uint64_t top_k = 4;
    float eps = 1e-5f;
    float lr = 1e-3f;
    float weight_decay = 0.01f;
    float grad_clip = 1.0f;
    int64_t warmup_steps = 500;
    int64_t total_steps = 10000;
    float min_lr = 1e-5f;
  };

  Hybrid(const Config& cfg);

  std::shared_ptr<core::Tensor> forward(
      const std::shared_ptr<core::Tensor>& input_ids) override;

  std::shared_ptr<core::Tensor> compute_loss(
      const std::shared_ptr<core::Tensor>& logits,
      const std::shared_ptr<core::Tensor>& target_ids) override;

  uint64_t num_parameters() const override { return num_params_; }
  std::string name() const override { return "Hybrid"; }
  std::vector<std::shared_ptr<core::Tensor>> parameters() override { return params_; }

  const Config& config() const { return cfg_; }

  float train_step(const std::shared_ptr<core::Tensor>& input_ids,
                   const std::shared_ptr<core::Tensor>& target_ids,
                   float current_lr);

 private:
  Config cfg_;
  uint64_t num_params_ = 0;
  std::vector<std::shared_ptr<core::Tensor>> params_;

  struct BlockParams {
    std::shared_ptr<core::Tensor> w_r_in, w_r_out;  // recurrent in/out
    std::shared_ptr<core::Tensor> decay;            // [d]
    std::shared_ptr<core::Tensor> w_q, w_k, w_v, w_o;  // attention
    std::shared_ptr<core::Tensor> w1, w2;           // FFN
    std::shared_ptr<core::Tensor> w_r1, w_r2;       // RMSNorm
  };
  std::shared_ptr<core::Tensor> w_emb_;
  std::shared_ptr<core::Tensor> w_pos_;
  std::vector<BlockParams> blocks_;
};

}  // namespace rak::models

#endif  // RAK_MODELS_HYBRID_H
