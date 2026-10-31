#include "rak/models/hybrid.h"
#include "rak/ops/embedding.h"
#include "rak/ops/local_attention.h"

#include <cmath>
#include <stdexcept>
#include <numeric>

namespace rak::models {

static std::shared_ptr<core::Tensor> make_shared_reshape(
    const std::shared_ptr<core::Tensor>& t,
    const std::vector<size_t>& shape) {
  return std::make_shared<core::Tensor>(t->reshape(shape));
}

static std::shared_ptr<core::Tensor> make_tensor_2d(
    size_t rows, size_t cols, std::vector<float> values, bool requires_grad) {
  std::vector<size_t> shape{rows, cols};
  auto t = std::make_shared<core::Tensor>(shape, values, requires_grad);
  t->requires_grad = requires_grad;
  t->is_leaf = requires_grad;
  if (requires_grad) t->grad.assign(values.size(), 0.0f);
  return t;
}

static std::shared_ptr<core::Tensor> make_tensor_1d(
    size_t n, std::vector<float> values, bool requires_grad) {
  std::vector<size_t> shape{n};
  auto t = std::make_shared<core::Tensor>(shape, values, requires_grad);
  t->requires_grad = requires_grad;
  t->is_leaf = requires_grad;
  if (requires_grad) t->grad.assign(values.size(), 0.0f);
  return t;
}

Hybrid::Hybrid(const Config& cfg) : cfg_(cfg) {
  {
    std::vector<float> values(cfg.vocab_size * cfg.d_model, 0.0f);
    w_emb_ = make_tensor_2d(cfg.vocab_size, cfg.d_model, values, true);
    params_.push_back(w_emb_);
  }
  {
    std::vector<float> values(cfg.max_seq * cfg.d_model, 0.0f);
    w_pos_ = make_tensor_2d(cfg.max_seq, cfg.d_model, values, true);
    params_.push_back(w_pos_);
  }

  for (uint64_t l = 0; l < cfg.n_layer; ++l) {
    BlockParams bp;
    auto rand = [&](size_t n) {
      std::vector<float> v(n);
      for (size_t i = 0; i < n; ++i)
        v[i] = 0.02f * (static_cast<float>(std::rand()) / RAND_MAX - 0.5f);
      return v;
    };

    // Recurrent scan in/out.
    {
      std::vector<float> values = rand(cfg.d_model * cfg.d_model);
      bp.w_r_in = make_tensor_2d(cfg.d_model, cfg.d_model, values, true);
    }
    {
      std::vector<float> values = rand(cfg.d_model * cfg.d_model);
      bp.w_r_out = make_tensor_2d(cfg.d_model, cfg.d_model, values, true);
    }
    {
      std::vector<float> values(cfg.d_model, 0.9f);
      bp.decay = make_tensor_1d(cfg.d_model, values, true);
    }
    params_.push_back(bp.w_r_in);
    params_.push_back(bp.w_r_out);
    params_.push_back(bp.decay);

    // Attention QKVO.
    {
      std::vector<float> values = rand(cfg.d_model * cfg.d_model);
      bp.w_q = make_tensor_2d(cfg.d_model, cfg.d_model, values, true);
    }
    {
      std::vector<float> values = rand(cfg.d_model * cfg.d_model);
      bp.w_k = make_tensor_2d(cfg.d_model, cfg.d_model, values, true);
    }
    {
      std::vector<float> values = rand(cfg.d_model * cfg.d_model);
      bp.w_v = make_tensor_2d(cfg.d_model, cfg.d_model, values, true);
    }
    {
      std::vector<float> values = rand(cfg.d_model * cfg.d_model);
      bp.w_o = make_tensor_2d(cfg.d_model, cfg.d_model, values, true);
    }
    params_.push_back(bp.w_q);
    params_.push_back(bp.w_k);
    params_.push_back(bp.w_v);
    params_.push_back(bp.w_o);

    // FFN.
    {
      std::vector<float> values = rand(cfg.d_model * cfg.d_ff);
      bp.w1 = make_tensor_2d(cfg.d_model, cfg.d_ff, values, true);
    }
    {
      std::vector<float> values = rand(cfg.d_ff * cfg.d_model);
      bp.w2 = make_tensor_2d(cfg.d_ff, cfg.d_model, values, true);
    }
    params_.push_back(bp.w1);
    params_.push_back(bp.w2);

    // RMSNorm.
    {
      std::vector<float> values(cfg.d_model, 1.0f);
      bp.w_r1 = make_tensor_1d(cfg.d_model, values, true);
    }
    {
      std::vector<float> values(cfg.d_model, 1.0f);
      bp.w_r2 = make_tensor_1d(cfg.d_model, values, true);
    }
    params_.push_back(bp.w_r1);
    params_.push_back(bp.w_r2);

    blocks_.push_back(std::move(bp));
  }

  num_params_ = 0;
  for (const auto& p : params_) num_params_ += p->numel();
}

std::shared_ptr<core::Tensor> Hybrid::forward(
    const std::shared_ptr<core::Tensor>& input_ids) {
  size_t batch = input_ids->shape[0];
  size_t seq = input_ids->shape[1];

  auto h = ops::embedding(input_ids, w_emb_);  // [batch, seq, d]

  for (const auto& bp : blocks_) {
    // h1 = RMSNorm(h)
    auto h1 = ops::rms_norm(h, bp.w_r1, cfg_.eps);

    // Recurrent branch: h1 @ w_r_in -> [batch, seq, d]
    auto h1_flat = make_shared_reshape(h1, {batch * seq, cfg_.d_model});
    auto rec_in = ops::gemm(h1_flat, bp.w_r_in);  // [batch*seq, d]
    auto rec_in_3d = make_shared_reshape(rec_in, {batch, seq, cfg_.d_model});
    auto rec_out = ops::recurrent_scan(rec_in_3d, bp.decay, "parallel", "forward");
    auto rec_out_flat = make_shared_reshape(rec_out, {batch * seq, cfg_.d_model});
    auto rec_branch = ops::gemm(rec_out_flat, bp.w_r_out);

    // Attention branch.
    auto q = ops::gemm(h1_flat, bp.w_q);   // [batch*seq, d]
    auto k = ops::gemm(h1_flat, bp.w_k);
    auto v = ops::gemm(h1_flat, bp.w_v);
    auto q_3d = make_shared_reshape(q, {batch, seq, cfg_.d_model});
    auto k_3d = make_shared_reshape(k, {batch, seq, cfg_.d_model});
    auto v_3d = make_shared_reshape(v, {batch, seq, cfg_.d_model});
    auto attn = ops::local_attention(q_3d, k_3d, v_3d, cfg_.window);  // [batch, seq, d]
    auto attn_flat = make_shared_reshape(attn, {batch * seq, cfg_.d_model});
    auto attn_out = ops::gemm(attn_flat, bp.w_o);  // [batch*seq, d]

    // Residual gabungan: h + rec_branch + attn_out
    auto h_flat = make_shared_reshape(h, {batch * seq, cfg_.d_model});
    auto h2 = ops::add(h_flat, rec_branch);
    h2 = ops::add(h2, attn_out);
    auto h2_3d = make_shared_reshape(h2, {batch, seq, cfg_.d_model});

    // RMSNorm
    auto h3 = ops::rms_norm(h2_3d, bp.w_r2, cfg_.eps);

    // FFN: [batch, seq, d] -> [batch*seq, d] -> w1 -> [batch*seq, d_ff] -> SiLU -> w2 -> [batch*seq, d]
    auto h3_flat = make_shared_reshape(h3, {batch * seq, cfg_.d_model});
    auto ff1 = ops::gemm(h3_flat, bp.w1);
    auto ff1_silu = ops::silu(ff1);
    auto ff2 = ops::gemm(ff1_silu, bp.w2);

    // Residual: h2 + ff2 -> [batch, seq, d]
    h = ops::add(h2, ff2);
    h = make_shared_reshape(h, {batch, seq, cfg_.d_model});
  }

  // Output head: h @ w_emb^T -> [batch, seq, vocab]
  auto h_flat = make_shared_reshape(h, {batch * seq, cfg_.d_model});
  auto logits_flat = ops::gemm(h_flat, w_emb_);
  auto logits = make_shared_reshape(logits_flat, {batch, seq, cfg_.vocab_size});

  return logits;
}

std::shared_ptr<core::Tensor> Hybrid::compute_loss(
    const std::shared_ptr<core::Tensor>& logits,
    const std::shared_ptr<core::Tensor>& target_ids) {
  size_t batch = logits->shape[0];
  size_t seq = logits->shape[1];
  size_t vocab = logits->shape[2];

  auto logits_flat = make_shared_reshape(logits, {batch * seq, vocab});
  auto log_probs = ops::log_softmax(logits_flat);

  auto loss = ops::mul(ops::mean(log_probs, -1),
      std::make_shared<core::Tensor>(
          std::vector<size_t>{1}, std::vector<float>{-1.0f}, false));

  return loss;
}

float Hybrid::train_step(
    const std::shared_ptr<core::Tensor>& input_ids,
    const std::shared_ptr<core::Tensor>& target_ids,
    float current_lr) {
  reset_tape();
  auto logits = forward(input_ids);
  auto loss = compute_loss(logits, target_ids);
  backward(loss);
  return loss->at(0);
}

}  // namespace rak::models
