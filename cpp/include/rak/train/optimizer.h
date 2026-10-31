#ifndef RAK_TRAIN_OPTIMIZER_H
#define RAK_TRAIN_OPTIMIZER_H

#include "rak/core/tensor.h"

#include <vector>
#include <string>
#include <cstdint>
#include <memory>

namespace rak::train {

// Jadwal learning rate: warmup linear lalu cosine decay (didefinisikan di akhir berkas).
class LRScheduler;

// Konfigurasi optimizer.
struct OptimizerConfig {
  float lr = 1e-3f;
  float weight_decay = 0.01f;
  float beta1 = 0.9f;
  float beta2 = 0.999f;
  float eps = 1e-8f;
  float grad_clip = 0.0f;  // 0 = tidak ada clipping
  int64_t warmup_steps = 0;   // 0 = tidak ada warmup
  int64_t total_steps = 0;    // 0 = pakai lr konstan (tanpa jadwal)
  float min_lr = 0.0f;        // learning rate minimum pada akhir cosine decay
};

// AdamW dengan jadwal learning rate (warmup + cosine).
class AdamW {
 public:
  AdamW(const OptimizerConfig& config);

  // Set daftar parameter (tensor leaf yang requires_grad).
  void add_parameter(std::shared_ptr<rak::core::Tensor> param);

  // Lakukan satu langkah optimasi.
  void step(int64_t global_step);

  // Gradient clipping berdasarkan norma global.
  void clip_gradients(float max_norm);

  // Hitung norma global gradien semua parameter.
  float global_grad_norm() const;

  // Learning rate saat ini (setelah panggil step terakhir).
  float current_lr() const { return current_lr_; }

  const OptimizerConfig& config() const { return config_; }

  // State untuk checkpoint/resume.
  struct State {
    std::vector<int64_t> step_counts;     // per parameter
    std::vector<std::vector<float>> m;    // first moment
    std::vector<std::vector<float>> v;    // second moment
    int64_t global_step = 0;
    float current_lr = 0.0f;
  };
  State save_state() const;
  void load_state(const State& state);

 private:
  OptimizerConfig config_;
  std::vector<std::shared_ptr<rak::core::Tensor>> params_;
  std::vector<int64_t> steps_;
  std::vector<std::vector<float>> m_;
  std::vector<std::vector<float>> v_;
  int64_t global_step_ = 0;
  float current_lr_ = 0.0f;
  std::unique_ptr<LRScheduler> scheduler_;  // jadwal LR (warmup + cosine) bila total_steps > 0

  float lr_at(int64_t step) const;
};

// Jadwal learning rate: warmup linear lalu cosine decay.
class LRScheduler {
 public:
  LRScheduler(float base_lr, int64_t warmup_steps, int64_t total_steps,
              float min_lr = 0.0f);

  float lr_at(int64_t step) const;

 private:
  float base_lr_;
  int64_t warmup_steps_;
  int64_t total_steps_;
  float min_lr_;
};

}  // namespace rak::train

#endif  // RAK_TRAIN_OPTIMIZER_H
