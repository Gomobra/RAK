#ifndef RAK_TRAIN_CHECKPOINT_H
#define RAK_TRAIN_CHECKPOINT_H

#include "rak/core/tensor.h"
#include "rak/train/optimizer.h"

#include <string>
#include <vector>
#include <cstdint>
#include <fstream>

namespace rak::train {

// Checkpoint: bobot, state optimizer, state RNG, posisi data.
// Format biner sederhana dengan header magic + versi + checksum CRC32.
//
// Determinisme resume:
//   - `seed` digunakan untuk menginisialisasi ulang mesin acak (RNG) saat melanjutkan
//     pelatihan dari checkpoint ini, sehingga urutan dropout / shuffling dapat direproduksi.
//   - `rng_state` menyimpan snapshot state RNG sebenarnya pada saat checkpoint disimpan,
//     sehingga resume dapat melanjutkan dari state yang sama tanpa mengandalkan seed saja.
//     Jika kosong, seed digunakan sebagai fallback inisialisasi RNG.
struct Checkpoint {
  int64_t global_step = 0;
  int64_t seed = 0;            // seed untuk inisialisasi RNG saat resume (dokumentasi determinisme)
  std::vector<uint8_t> rng_state;  // snapshot state RNG pada waktu disimpan (boleh kosong)
  std::vector<std::string> param_names;
  std::vector<std::vector<float>> param_values;
  std::vector<std::vector<float>> param_grads;
  AdamW::State optimizer_state;
  size_t data_position = 0;    // indeks batch/shard saat checkpoint disimpan
};

// Simpan checkpoint ke berkas.
void save_checkpoint(const std::string& path, const Checkpoint& ckpt);

// Muat checkpoint dari berkas.
Checkpoint load_checkpoint(const std::string& path);

// Hitung CRC32 sederhana untuk integritas.
uint32_t compute_crc32(const std::vector<uint8_t>& data);

}  // namespace rak::train

#endif  // RAK_TRAIN_CHECKPOINT_H
