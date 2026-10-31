#include "rak/train/optimizer.h"

#include <cmath>
#include <stdexcept>
#include <numeric>
#include <algorithm>
#include <memory>

namespace rak::train {

AdamW::AdamW(const OptimizerConfig& config) : config_(config) {
  // Buat jadwal LR bila total_steps > 0 (warmup linear lalu cosine decay).
  if (config.total_steps > 0) {
    scheduler_ = std::make_unique<LRScheduler>(config.lr, config.warmup_steps,
                                               config.total_steps, config.min_lr);
  }
}

void AdamW::add_parameter(std::shared_ptr<rak::core::Tensor> param) {
  if (!param) throw std::invalid_argument("AdamW: parameter null");
  if (!param->requires_grad) {
    throw std::invalid_argument("AdamW: parameter harus requires_grad=true");
  }
  params_.push_back(std::move(param));
  steps_.push_back(0);
  m_.emplace_back(params_.back()->numel(), 0.0f);
  v_.emplace_back(params_.back()->numel(), 0.0f);
}

float AdamW::lr_at(int64_t step) const {
  // Jika total_steps disetel (dan ada scheduler), gunakan jadwal warmup linear +
  // cosine decay. Lainnya pakai lr konstan.
  if (scheduler_) return scheduler_->lr_at(step);
  return config_.lr;
}

void AdamW::step(int64_t global_step) {
  global_step_ = global_step;
  current_lr_ = lr_at(global_step);

  // Hitung norma global untuk clipping (jika diaktifkan).
  float global_norm = 0.0f;
  if (config_.grad_clip > 0.0f) {
    for (size_t i = 0; i < params_.size(); ++i) {
      auto& p = params_[i];
      for (size_t j = 0; j < p->grad.size(); ++j) {
        global_norm += p->grad[j] * p->grad[j];
      }
    }
    global_norm = std::sqrt(global_norm);
  }

  float clip_coef = 1.0f;
  if (config_.grad_clip > 0.0f && global_norm > config_.grad_clip) {
    clip_coef = config_.grad_clip / (global_norm + config_.eps);
  }

  for (size_t i = 0; i < params_.size(); ++i) {
    auto& p = params_[i];
    size_t n = p->numel();
    steps_[i] += 1;
    int64_t t = steps_[i];

    float lr = current_lr_;
    float bias_correction1 = 1.0f - std::pow(config_.beta1, t);
    float bias_correction2 = 1.0f - std::pow(config_.beta2, t);

    for (size_t j = 0; j < n; ++j) {
      float g = p->grad[j] * clip_coef;

      // Weight decay (decoupled, AdamW).
      if (config_.weight_decay > 0.0f) {
        g += config_.weight_decay * p->ptr[j];
      }

      m_[i][j] = config_.beta1 * m_[i][j] + (1.0f - config_.beta1) * g;
      v_[i][j] = config_.beta2 * v_[i][j] + (1.0f - config_.beta2) * (g * g);

      float m_hat = m_[i][j] / bias_correction1;
      float v_hat = v_[i][j] / bias_correction2;

      p->ptr[j] -= lr * m_hat / (std::sqrt(v_hat) + config_.eps);
    }
  }
}

void AdamW::clip_gradients(float max_norm) {
  float norm = global_grad_norm();
  if (norm > max_norm && norm > 0.0f) {
    float coef = max_norm / norm;
    for (auto& p : params_) {
      for (auto& g : p->grad) g *= coef;
    }
  }
}

float AdamW::global_grad_norm() const {
  float norm = 0.0f;
  for (auto& p : params_) {
    for (auto& g : p->grad) norm += g * g;
  }
  return std::sqrt(norm);
}

AdamW::State AdamW::save_state() const {
  State st;
  st.global_step = global_step_;
  st.current_lr = current_lr_;
  st.step_counts = steps_;
  st.m = m_;
  st.v = v_;
  return st;
}

void AdamW::load_state(const State& state) {
  global_step_ = state.global_step;
  current_lr_ = state.current_lr;
  steps_ = state.step_counts;
  m_ = state.m;
  v_ = state.v;
}

LRScheduler::LRScheduler(float base_lr, int64_t warmup_steps, int64_t total_steps,
                         float min_lr)
    : base_lr_(base_lr), warmup_steps_(warmup_steps), total_steps_(total_steps), min_lr_(min_lr) {}

float LRScheduler::lr_at(int64_t step) const {
  // Tanpa jadwal (total_steps == 0): learning rate konstan.
  if (total_steps_ == 0) return base_lr_;
  if (step <= warmup_steps_) {
    if (warmup_steps_ == 0) return base_lr_;
    return base_lr_ * static_cast<float>(step) / static_cast<float>(warmup_steps_);
  }
  if (step >= total_steps_) return min_lr_;
  float progress = static_cast<float>(step - warmup_steps_) /
                   static_cast<float>(total_steps_ - warmup_steps_);
  return min_lr_ + 0.5f * (base_lr_ - min_lr_) *
             (1.0f + std::cos(M_PI * progress));
}

}  // namespace rak::train
