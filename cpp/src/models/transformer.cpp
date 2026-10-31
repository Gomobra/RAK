#include "rak/models/transformer.h"
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

Transformer::Transformer(const Config& cfg) : cfg_(cfg) {
  {
    std::vector<size_t> shape{cfg.vocab_size, cfg.d_model};
    std::vector<float> values(cfg.vocab_size * cfg.d_model, 0.0f);
    w_emb_ = std::make_shared<core::Tensor>(shape, values, true);
    w_emb_->requires_grad = true;
    w_emb_->is_leaf = true;
    w_emb_->grad.assign(cfg.vocab_size * cfg.d_model, 0.0f);
    params_.push_back(w_emb_);
  }
  {
    std::vector<size_t> shape{cfg.max_seq, cfg.d_model};
    std::vector<float> values(cfg.max_seq * cfg.d_model, 0.0f);
    w_pos_ = std::make_shared<core::Tensor>(shape, values, true);
    w_pos_->requires_grad = true;
    w_pos_->is_leaf = true;
    w_pos_->grad.assign(cfg.max_seq * cfg.d_model, 0.0f);
    params_.push_back(w_pos_);
  }

  for (uint64_t l = 0; l < cfg.n_layer; ++l) {
    BlockParams bp;
    auto rand_float = [&](size_t n) {
      std::vector<float> v(n);
      for (size_t i = 0; i < n; ++i)
        v[i] = 0.02f * (static_cast<float>(std::rand()) / RAND_MAX - 0.5f);
      return v;
    };

    // MHA: QKV = d_model x (3*d_model), output = d_model x d_model
    {
      std::vector<size_t> shape{cfg.d_model, 3 * cfg.d_model};
      std::vector<float> values = rand_float(cfg.d_model * 3 * cfg.d_model);
      bp.w_q = std::make_shared<core::Tensor>(shape, values, true);
    }
    {
      std::vector<size_t> shape{cfg.d_model, cfg.d_model};
      std::vector<float> values = rand_float(cfg.d_model * cfg.d_model);
      bp.w_o = std::make_shared<core::Tensor>(shape, values, true);
    }
    bp.w_q->requires_grad = true; bp.w_q->is_leaf = true;
    bp.w_o->requires_grad = true; bp.w_o->is_leaf = true;
    bp.w_q->grad.assign(cfg.d_model * 3 * cfg.d_model, 0.0f);
    bp.w_o->grad.assign(cfg.d_model * cfg.d_model, 0.0f);
    params_.push_back(bp.w_q); params_.push_back(bp.w_o);

    // FFN: d_model -> d_ff -> d_model
    {
      std::vector<size_t> shape{cfg.d_model, cfg.d_ff};
      std::vector<float> values = rand_float(cfg.d_model * cfg.d_ff);
      bp.w1 = std::make_shared<core::Tensor>(shape, values, true);
    }
    {
      std::vector<size_t> shape{cfg.d_ff, cfg.d_model};
      std::vector<float> values = rand_float(cfg.d_ff * cfg.d_model);
      bp.w2 = std::make_shared<core::Tensor>(shape, values, true);
    }
    bp.w1->requires_grad = true; bp.w1->is_leaf = true;
    bp.w2->requires_grad = true; bp.w2->is_leaf = true;
    bp.w1->grad.assign(cfg.d_model * cfg.d_ff, 0.0f);
    bp.w2->grad.assign(cfg.d_ff * cfg.d_model, 0.0f);
    params_.push_back(bp.w1); params_.push_back(bp.w2);

    // RMSNorm weight (dua kali per blok)
    {
      std::vector<size_t> shape{cfg.d_model};
      std::vector<float> values(cfg.d_model, 1.0f);
      bp.w_r1 = std::make_shared<core::Tensor>(shape, values, true);
    }
    {
      std::vector<size_t> shape{cfg.d_model};
      std::vector<float> values(cfg.d_model, 1.0f);
      bp.w_r2 = std::make_shared<core::Tensor>(shape, values, true);
    }
    bp.w_r1->requires_grad = true; bp.w_r1->is_leaf = true;
    bp.w_r2->requires_grad = true; bp.w_r2->is_leaf = true;
    bp.w_r1->grad.assign(cfg.d_model, 0.0f);
    bp.w_r2->grad.assign(cfg.d_model, 0.0f);
    params_.push_back(bp.w_r1); params_.push_back(bp.w_r2);

    blocks_.push_back(std::move(bp));
  }

  num_params_ = 0;
  for (const auto& p : params_) num_params_ += p->numel();
}

std::shared_ptr<core::Tensor> Transformer::forward(
    const std::shared_ptr<core::Tensor>& input_ids) {
  size_t batch = input_ids->shape[0];
  size_t seq = input_ids->shape[1];

  auto h = ops::embedding(input_ids, w_emb_);  // [batch, seq, d]

  for (const auto& bp : blocks_) {
    // h1 = RMSNorm(h)
    auto h1 = ops::rms_norm(h, bp.w_r1, cfg_.eps);

    // Prototype self-attention: gunakan h1 sebagai Q, K, V
    auto q_h = h1;
    auto k_h = h1;
    auto v_h = h1;

    // local_attention dengan window = seq (attention penuh)
    auto attn = ops::local_attention(q_h, k_h, v_h, seq);  // [batch, seq, d]

    // Proyeksi output attn: attn @ w_o -> [batch, seq, d]
    auto attn_out = ops::gemm(attn, bp.w_o);

    // Residual: h + attn_out
    auto h2 = ops::add(h, attn_out);

    // RMSNorm
    auto h3 = ops::rms_norm(h2, bp.w_r2, cfg_.eps);

    // FFN: h3 @ w1 -> [batch, seq, d_ff], SiLU, @ w2 -> [batch, seq, d]
    auto ff1 = ops::gemm(h3, bp.w1);
    auto ff1_silu = ops::silu(ff1);
    auto ff2 = ops::gemm(ff1_silu, bp.w2);

    // Residual
    h = ops::add(h2, ff2);
  }

  // Output head: h @ w_emb^T (tied) -> [batch, seq, vocab]
  auto h_flat = make_shared_reshape(h, {batch * seq, cfg_.d_model});
  auto logits_flat = ops::gemm(h_flat, w_emb_);  // [batch*seq, vocab]
  auto logits = make_shared_reshape(logits_flat, {batch, seq, cfg_.vocab_size});

  return logits;
}

std::shared_ptr<core::Tensor> Transformer::compute_loss(
    const std::shared_ptr<core::Tensor>& logits,
    const std::shared_ptr<core::Tensor>& target_ids) {
  size_t batch = logits->shape[0];
  size_t seq = logits->shape[1];
  size_t vocab = logits->shape[2];

  // Reshape logits ke [batch*seq, vocab], target ke [batch*seq]
  auto logits_flat = make_shared_reshape(logits, {batch * seq, vocab});
  auto target_flat = make_shared_reshape(target_ids, {batch * seq});

  // log_softmax di axis terakhir (vocab)
  auto log_probs = ops::log_softmax(logits_flat);  // [batch*seq, vocab]

  // Untuk Fase 1 prototype: gunakan mean negatif log_probs sebagai proxy loss.
  // Sparse cross-entropy sebenarnya membutuhkan op gather; diimplementasikan
  // lengkap setelah tensor gather tersedia.
  auto loss = ops::mul(ops::mean(log_probs, -1), std::make_shared<core::Tensor>(
      std::vector<size_t>{1}, std::vector<float>{-1.0f}, false));

  return loss;
}

float Transformer::train_step(
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
