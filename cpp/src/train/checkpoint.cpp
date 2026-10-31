#include "rak/train/checkpoint.h"

#include <stdexcept>
#include <fstream>
#include <cstring>
#include <algorithm>
#include <numeric>

namespace rak::train {

static const char kCkptMagic[4] = {'R', 'A', 'K', 'C'};
static const int kCkptVersion = 1;

uint32_t compute_crc32(const std::vector<uint8_t>& data) {
  // CRC32 sederhana (tabel-bangun).
  static uint32_t table[256];
  static bool init = false;
  if (!init) {
    for (uint32_t i = 0; i < 256; ++i) {
      uint32_t c = i;
      for (int k = 0; k < 8; ++k) {
        c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
      }
      table[i] = c;
    }
    init = true;
  }
  uint32_t crc = 0xFFFFFFFFu;
  for (uint8_t b : data) crc = table[(crc ^ b) & 0xFF] ^ (crc >> 8);
  return crc ^ 0xFFFFFFFFu;
}

void save_checkpoint(const std::string& path, const Checkpoint& ckpt) {
  std::ofstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("save_checkpoint: tidak dapat membuka " + path);

  // Header: magic + versi.
  f.write(kCkptMagic, 4);
  int32_t version = kCkptVersion;
  f.write(reinterpret_cast<const char*>(&version), sizeof(version));

  // Metadata.
  f.write(reinterpret_cast<const char*>(&ckpt.global_step), sizeof(ckpt.global_step));
  f.write(reinterpret_cast<const char*>(&ckpt.seed), sizeof(ckpt.seed));
  int32_t n_param = static_cast<int32_t>(ckpt.param_names.size());
  f.write(reinterpret_cast<const char*>(&n_param), sizeof(n_param));

  for (size_t i = 0; i < ckpt.param_names.size(); ++i) {
    int32_t name_len = static_cast<int32_t>(ckpt.param_names[i].size());
    f.write(reinterpret_cast<const char*>(&name_len), sizeof(name_len));
    f.write(ckpt.param_names[i].data(), name_len);
    int64_t nval = static_cast<int64_t>(ckpt.param_values[i].size());
    f.write(reinterpret_cast<const char*>(&nval), sizeof(nval));
    f.write(reinterpret_cast<const char*>(ckpt.param_values[i].data()), nval * sizeof(float));
    f.write(reinterpret_cast<const char*>(ckpt.param_grads[i].data()), nval * sizeof(float));
  }

  // State optimizer.
  int64_t gs = ckpt.optimizer_state.global_step;
  f.write(reinterpret_cast<const char*>(&gs), sizeof(gs));
  float clr = ckpt.optimizer_state.current_lr;
  f.write(reinterpret_cast<const char*>(&clr), sizeof(clr));
  int32_t nstep = static_cast<int32_t>(ckpt.optimizer_state.step_counts.size());
  f.write(reinterpret_cast<const char*>(&nstep), sizeof(nstep));
  f.write(reinterpret_cast<const char*>(ckpt.optimizer_state.step_counts.data()),
          nstep * sizeof(int64_t));
  int32_t nm = static_cast<int32_t>(ckpt.optimizer_state.m.size());
  f.write(reinterpret_cast<const char*>(&nm), sizeof(nm));
  for (const auto& m : ckpt.optimizer_state.m) {
    int64_t nv = static_cast<int64_t>(m.size());
    f.write(reinterpret_cast<const char*>(&nv), sizeof(nv));
    f.write(reinterpret_cast<const char*>(m.data()), nv * sizeof(float));
  }
  int32_t nv = static_cast<int32_t>(ckpt.optimizer_state.v.size());
  f.write(reinterpret_cast<const char*>(&nv), sizeof(nv));
  for (const auto& v : ckpt.optimizer_state.v) {
    int64_t nvv = static_cast<int64_t>(v.size());
    f.write(reinterpret_cast<const char*>(&nvv), sizeof(nvv));
    f.write(reinterpret_cast<const char*>(v.data()), nvv * sizeof(float));
  }

  int64_t dp = static_cast<int64_t>(ckpt.data_position);
  f.write(reinterpret_cast<const char*>(&dp), sizeof(dp));

  // State RNG (snapshot untuk resume deterministik; boleh kosong).
  int64_t n_rng = static_cast<int64_t>(ckpt.rng_state.size());
  f.write(reinterpret_cast<const char*>(&n_rng), sizeof(n_rng));
  if (n_rng > 0) {
    f.write(reinterpret_cast<const char*>(ckpt.rng_state.data()), n_rng);
  }

  // Checksum di akhir: hitung CRC32 dari seluruh payload, lalu tambahkan 4 byte.
  f.flush();
  f.close();

  // Baca ulang file, hitung CRC, tambahkan di akhir.
  std::ifstream rf(path, std::ios::binary);
  std::vector<uint8_t> buffer((std::istreambuf_iterator<char>(rf)),
                               std::istreambuf_iterator<char>());
  uint32_t crc = compute_crc32(buffer);
  rf.close();

  std::ofstream wf(path, std::ios::binary | std::ios::app);
  uint32_t crc_le = crc;
  wf.write(reinterpret_cast<const char*>(&crc_le), sizeof(crc_le));
}

Checkpoint load_checkpoint(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("load_checkpoint: tidak dapat membuka " + path);

  // Baca seluruh file ke buffer untuk verifikasi CRC32 terhadap payload
  // (seluruh isi file tanpa 4 byte CRC terakhir).
  std::streamsize file_size = f.tellg();
  f.seekg(0, std::ios::beg);
  if (file_size < static_cast<std::streamsize>(4 + 4 + 4 + 4)) {
    throw std::runtime_error("load_checkpoint: file terlalu pendek untuk berisi header + CRC");
  }

  std::vector<uint8_t> file_data(static_cast<size_t>(file_size));
  if (!f.read(reinterpret_cast<char*>(file_data.data()), file_size)) {
    throw std::runtime_error("load_checkpoint: gagal membaca isi file " + path);
  }

  // Ambil CRC yang disimpan (4 byte terakhir) dan payload (seluruhnya kecuali CRC).
  uint32_t stored_crc = 0;
  std::memcpy(&stored_crc, file_data.data() + file_size - 4, 4);
  std::vector<uint8_t> payload(file_data.begin(),
                               file_data.end() - 4);
  uint32_t calc_crc = compute_crc32(payload);
  if (calc_crc != stored_crc) {
    throw std::runtime_error("load_checkpoint: CRC32 tidak cocok (file rusak atau terputih): "
                             + std::to_string(calc_crc) + " != " + std::to_string(stored_crc));
  }

  // Parse payload dengan cursor + pemeriksaan batas.
  size_t pos = 0;
  auto read_bytes = [&](void* out, size_t n) {
    if (pos + n > payload.size()) {
      throw std::runtime_error("load_checkpoint: file terpotong (payload tidak lengkap)");
    }
    std::memcpy(out, payload.data() + pos, n);
    pos += n;
  };
  auto read_str = [&](std::string& s) {
    int32_t len = 0;
    read_bytes(&len, sizeof(len));
    if (len < 0 || static_cast<size_t>(len) > payload.size() - pos) {
      throw std::runtime_error("load_checkpoint: panjang string tidak valid");
    }
    s.resize(static_cast<size_t>(len));
    read_bytes(&s[0], static_cast<size_t>(len));
  };

  Checkpoint ckpt;

  char magic[4];
  read_bytes(magic, 4);
  if (std::memcmp(magic, kCkptMagic, 4) != 0) {
    throw std::runtime_error("load_checkpoint: magic tidak cocok (bukan file checkpoint RAK)");
  }
  int32_t version = 0;
  read_bytes(&version, sizeof(version));
  if (version != kCkptVersion) {
    throw std::runtime_error("load_checkpoint: versi tidak didukung: " +
                             std::to_string(version));
  }

  read_bytes(&ckpt.global_step, sizeof(ckpt.global_step));
  read_bytes(&ckpt.seed, sizeof(ckpt.seed));
  int32_t n_param = 0;
  read_bytes(&n_param, sizeof(n_param));
  if (n_param < 0) {
    throw std::runtime_error("load_checkpoint: jumlah parameter tidak valid");
  }

  ckpt.param_names.resize(static_cast<size_t>(n_param));
  ckpt.param_values.resize(static_cast<size_t>(n_param));
  ckpt.param_grads.resize(static_cast<size_t>(n_param));
  for (int32_t i = 0; i < n_param; ++i) {
    read_str(ckpt.param_names[i]);
    int64_t nval = 0;
    read_bytes(&nval, sizeof(nval));
    if (nval < 0 || static_cast<size_t>(nval) > (payload.size() - pos) / sizeof(float)) {
      throw std::runtime_error("load_checkpoint: jumlah nilai parameter tidak valid");
    }
    ckpt.param_values[i].resize(static_cast<size_t>(nval));
    ckpt.param_grads[i].resize(static_cast<size_t>(nval));
    read_bytes(ckpt.param_values[i].data(), static_cast<size_t>(nval) * sizeof(float));
    read_bytes(ckpt.param_grads[i].data(), static_cast<size_t>(nval) * sizeof(float));
  }

  int64_t gs = 0;
  read_bytes(&gs, sizeof(gs));
  ckpt.optimizer_state.global_step = gs;
  float clr = 0.0f;
  read_bytes(&clr, sizeof(clr));
  ckpt.optimizer_state.current_lr = clr;
  int32_t nstep = 0;
  read_bytes(&nstep, sizeof(nstep));
  if (nstep < 0) throw std::runtime_error("load_checkpoint: jumlah step_counts tidak valid");
  ckpt.optimizer_state.step_counts.resize(static_cast<size_t>(nstep));
  read_bytes(ckpt.optimizer_state.step_counts.data(),
             static_cast<size_t>(nstep) * sizeof(int64_t));
  int32_t nm = 0;
  read_bytes(&nm, sizeof(nm));
  if (nm < 0) throw std::runtime_error("load_checkpoint: jumlah m tidak valid");
  ckpt.optimizer_state.m.resize(static_cast<size_t>(nm));
  for (int32_t i = 0; i < nm; ++i) {
    int64_t nv = 0;
    read_bytes(&nv, sizeof(nv));
    if (nv < 0 || static_cast<size_t>(nv) > (payload.size() - pos) / sizeof(float)) {
      throw std::runtime_error("load_checkpoint: panjang m tidak valid");
    }
    ckpt.optimizer_state.m[i].resize(static_cast<size_t>(nv));
    read_bytes(ckpt.optimizer_state.m[i].data(), static_cast<size_t>(nv) * sizeof(float));
  }
  int32_t nvv = 0;
  read_bytes(&nvv, sizeof(nvv));
  if (nvv < 0) throw std::runtime_error("load_checkpoint: jumlah v tidak valid");
  ckpt.optimizer_state.v.resize(static_cast<size_t>(nvv));
  for (int32_t i = 0; i < nvv; ++i) {
    int64_t nvv2 = 0;
    read_bytes(&nvv2, sizeof(nvv2));
    if (nvv2 < 0 || static_cast<size_t>(nvv2) > (payload.size() - pos) / sizeof(float)) {
      throw std::runtime_error("load_checkpoint: panjang v tidak valid");
    }
    ckpt.optimizer_state.v[i].resize(static_cast<size_t>(nvv2));
    read_bytes(ckpt.optimizer_state.v[i].data(), static_cast<size_t>(nvv2) * sizeof(float));
  }

  int64_t dp = 0;
  read_bytes(&dp, sizeof(dp));
  ckpt.data_position = static_cast<size_t>(dp);

  // State RNG (snapshot untuk resume deterministik).
  int64_t n_rng = 0;
  read_bytes(&n_rng, sizeof(n_rng));
  if (n_rng < 0 || static_cast<size_t>(n_rng) > payload.size() - pos) {
    throw std::runtime_error("load_checkpoint: panjang rng_state tidak valid");
  }
  ckpt.rng_state.resize(static_cast<size_t>(n_rng));
  read_bytes(ckpt.rng_state.data(), static_cast<size_t>(n_rng));

  return ckpt;
}

}  // namespace rak::train
