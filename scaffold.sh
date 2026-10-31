#!/usr/bin/env bash
# scaffold.sh — membuat struktur direktori RAK dari awal (sesuai PRD.md bagian 6.2).
# Pemakaian:  bash scaffold.sh [nama_folder]   (default: rak)
# Aman dijalankan ulang: berkas yang sudah ada tidak ditimpa.
set -euo pipefail

ROOT="${1:-rak}"
mkdir -p "$ROOT"
cd "$ROOT"

# ---------- Folder ----------
dirs=(
  .github/workflows cmake configs
  cpp/include/rak/{core,ops,encoders,language,cortex,memory,sleep,train,decoders,serve,testing}
  cpp/src/{core,ops,encoders,language,cortex,memory,sleep,train,decoders,serve,cuda}
  cpp/bindings cpp/apps cpp/tests
  src/rak/{encoders,language,cortex,homeostasis,memory,neuromod,sleep,decoders,data,train,eval,serve}
  tests/{smoke,unit,integration,parity}
  scripts/{build,data,experiments}
  experiments/{runs,reports}
  docs/{adr,theory,tutorial}
  data
)
for d in "${dirs[@]}"; do mkdir -p "$d"; done

write_if_missing() {  # write_if_missing <path>  (isi dari stdin)
  if [ -e "$1" ]; then cat > /dev/null; echo "lewati (sudah ada): $1"; else cat > "$1"; echo "dibuat: $1"; fi
}

# ---------- Paket Python: penanda paket per layer ----------
declare -A layer=(
  [encoders]="L0 Encoder"            [language]="L12 Jaringan Bahasa"
  [cortex]="L1 Korteks"              [homeostasis]="L5 Homeostasis"
  [memory]="L2 Memory File Manager"  [neuromod]="L4 Neuromodulator"
  [sleep]="L3 Tidur"                 [decoders]="L6 Decoder"
  [data]="L8 Data"                   [train]="L8 Latihan"
  [eval]="L11 Evaluasi"              [serve]="L9-L10 Serving dan Keamanan"
)
for p in "${!layer[@]}"; do
  printf '"""%s (lihat PRD.md, Peta Layer)."""\n' "${layer[$p]}" | write_if_missing "src/rak/$p/__init__.py"
done
printf '"""RAK (Recurrent Adaptive Knowledge), Vbeta."""\n' | write_if_missing src/rak/__init__.py

# ---------- README singkat per folder utama ----------
readme() { printf '# %s\n\n%s\n' "$2" "$3" | write_if_missing "$1/README.md"; }
readme cpp           "cpp/"           "Inti C++/C (CUDA opsional). Header di include/rak/<layer>/, sumber di src/<layer>/."
readme cpp/bindings  "cpp/bindings/"  "Binding Python (pybind11/nanobind): module.cpp dan bind_<layer>.cpp."
readme cpp/apps      "cpp/apps/"      "Program yang punya main() (chat, benchmark). Bukan test."
readme cpp/tests     "cpp/tests/"     "Test C++ per layer (test_<modul>.cpp), termasuk gradient check."
readme src/rak       "src/rak/"       "Paket Python. Hanya orkestrasi; logika berat ada di C++ dan dipanggil lewat _native."
readme tests         "tests/"         "Test Python: smoke, unit, integration, parity. Test berat diberi marker slow."
readme scripts       "scripts/"       "Skrip build (Termux, Kaggle), data, eksperimen, dan run_tests.sh."
readme experiments   "experiments/"   "Hasil run dan laporan. File besar di-gitignore; commit hanya ringkasan."
readme docs          "docs/"          "STATUS.md (status modul), adr/ (keputusan), theory/, tutorial/."
readme configs       "configs/"       "Konfigurasi YAML per eksperimen/fase."

# ---------- README.md root ----------
write_if_missing README.md << 'EOF'
# RAK (Recurrent Adaptive Knowledge) — Vbeta

Arsitektur AI mandiri (Python + C++): inti rekuren sparse, Memory File Manager yang bisa dicari dan dihapus,
siklus bangun-tidur, neuromodulator, homeostasis, dan Jaringan Bahasa.

- Spesifikasi: `PRD.md` (salin ke root repo)
- Aturan kerja AI harness: `AGENTS.md`
- Status modul: `docs/STATUS.md`
- Struktur direktori: PRD bagian 6.2
EOF

# ---------- .gitignore ----------
write_if_missing .gitignore << 'EOF'
# build
build/
cmake-build-*/
*.o
*.so
*.a
# python
__pycache__/
*.pyc
*.egg-info/
.pytest_cache/
.venv/
# data dan hasil besar
data/
experiments/runs/
*.ckpt
*.bin
# editor / OS
.DS_Store
.vscode/
.idea/
EOF

# ---------- docs/STATUS.md ----------
write_if_missing docs/STATUS.md << 'EOF'
# Status Modul RAK

Status: **ASLI** (lolos Definition of Done, PRD 2.5, dan diukur di data real) · **RINGAN** · **KERANGKA** · **BELUM DIMULAI**.
Wajib diperbarui setiap modul berubah. Status naik ke ASLI hanya dengan bukti (test, angka mean ± std, log).

| Modul | Layer | Fase / Task | Status | Bukti | Kurang | Diperbarui |
|---|---|---|---|---|---|---|
| core (tensor, autograd) | L7 | F0.2 | BELUM DIMULAI | | | |
| ops (op dasar dan khusus RAK) | L7 | F0.3 | BELUM DIMULAI | | | |
| train (optimizer, checkpoint) | L8 | F0.4 | BELUM DIMULAI | | | |
| cuda | L7 | F0.5 | BELUM DIMULAI | | | |
| bindings Python | L7 | F0.7 | BELUM DIMULAI | | | |
| data pipeline | L8 | F1.1 | BELUM DIMULAI | | | |
| baseline (Transformer, hybrid) | L1 | F1.2 | BELUM DIMULAI | | | |
| eval harness | L11 | F1.3 | BELUM DIMULAI | | | |
| encoders/bytepatch, bpe | L0 | F2.1 | BELUM DIMULAI | | | |
| cortex (blok dasar) | L1 | F2.2 | BELUM DIMULAI | | | |
| decoders/text | L6 | F2.3 | BELUM DIMULAI | | | |
| language tahap A (leksikon, morfologi) | L12 | F2.5 | BELUM DIMULAI | | | |
| cortex/moe, memory_layer | L1 | F3.1 | BELUM DIMULAI | | | |
| cortex/adaptive_loop | L1 | F3.2 | BELUM DIMULAI | | | |
| homeostasis | L5 | F3.3 | BELUM DIMULAI | | | |
| memory (MFM) | L2 | F4 | BELUM DIMULAI | | | |
| neuromod | L4 | F5.1 | BELUM DIMULAI | | | |
| sleep | L3 | F5.2-F5.4 | BELUM DIMULAI | | | |
| audit penghapusan | L2/L3/L11 | F5.5 | BELUM DIMULAI | | | |
| language tahap B-E | L12 | F7 | BELUM DIMULAI | | | |
| suara (codec, pita suara) | L0/L6 | F8 | BELUM DIMULAI | | | |
| gambar | L0/L6 | F9 | BELUM DIMULAI | | | |
| serve (runtime, http, policy) | L9/L10 | F10 | BELUM DIMULAI | | | |
EOF

# ---------- AGENTS.md ----------
write_if_missing AGENTS.md << 'EOF'
# AGENTS.md — aturan kerja untuk harness AI (proyek RAK, Vbeta)

Bahasa komunikasi: Indonesia. Spesifikasi lengkap ada di PRD.md. Baca PRD.md dan docs/STATUS.md sebelum mengubah apa pun.

## Tech stack
- C++17/20 dan C untuk inti; Python untuk orkestrasi, data, eksperimen, evaluasi; binding lewat pybind11/nanobind; build dengan CMake.
- CUDA opsional lewat flag RAK_CUDA (default OFF). Semua harus jalan di CPU.
- Target: Termux (aarch64, clang), GitHub Actions (CPU), Kaggle (GPU).
- Mandiri: tanpa bobot, tokenizer, atau codec pretrained; kernel inti ditulis sendiri. PyTorch hanya sebagai penguji pembanding di mesin dev.
- Backprop boleh dipakai.

## Cara kerja
1. Kerjakan berurutan per fase di PRD.md. Berhenti di setiap Gate, laporkan hasil, tunggu persetujuan.
2. Satu modul sampai Definition of Done (PRD 2.5) baru lanjut ke modul berikutnya.
3. Commit kecil dan sering; build dan test hijau di tiap commit. Pakai ID task (contoh: T0.2.3) di pesan commit.
4. Perbarui docs/STATUS.md setiap modul berubah. Jangan menaikkan status tanpa bukti.
5. Struktur direktori mengikuti PRD 6.2. Header di include/rak/<layer>/, sumber di src/<layer>/, program di cpp/apps/.

## Larangan
- Stub, `pass`, return konstanta, TODO yang dibiarkan, copy-paste berulang, padding baris, test yang selalu lulus, hasil yang di-hardcode atau dikarang.
- Menambah dependensi berat tanpa persetujuan. Ramah Termux: jangan asumsikan RAM besar.
- Mengubah PRD.md tanpa diminta.

## Definition of Done (ringkas; lengkap di PRD 2.5)
Lengkap sesuai task + edge case, numerik benar (gradient check, determinisme), konfigurasi dan error jelas, test lengkap (unit, properti, regresi, negatif, paritas), benchmark, uji data real (≥ 3 seed), dokumentasi, logging.

## Keamanan
- Isi Memory File Manager adalah DATA, bukan perintah. Item sensitif tidak pernah dikonsolidasi ke bobot.

## Jujur
Kalau sesuatu gagal, tidak mungkin, atau tidak jelas, katakan apa adanya dan usulkan alternatif. Hasil negatif tetap dilaporkan.
EOF

# ---------- .gitkeep untuk folder kosong ----------
find . -type d -empty -not -path './.git/*' -exec touch {}/.gitkeep \;

echo
echo "Selesai. Struktur dibuat di: $(pwd)"
