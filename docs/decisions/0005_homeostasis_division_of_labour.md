# ADR-0005: Pembagian kerja tiga mekanisme homeostasis

- Status: Accepted
- Date: 2026-10-15
- Affects: T1B, gate test Phase 2

## Konteks

Phase 2 mengimplementasikan tiga mekanisme homeostasis. Pertanyaan yang
muncul saat menulis test: **apakah ketiganya melakukan hal yang sama?**

Awalnya diasumsikan iya — "homeostasis = bikin neuron lebih stabil".
Pengujian menunjukkan itu salah.

## Data

Diuji di `tests/test_homeostatic_gate.py` dan `test_homeostasis.py`.

### Gate test (arsitektur recurrent + STDP, η+ = 2e-3, η− = 1e-4)

| Konfigurasi | Drive akhir | w_max | Status |
|---|---|---|---|
| STDP saja | 2047.6 | 1.00 | **SEMUA synapse saturasi** |
| STDP + homeostasis | 135.3 | 0.15 | **Tertahan, w_max jauh dari batas** |

Drive naik: 328 → 467 → 678 → 1049 → 1611 → 2048 (naik monoton sampai
plafon keras `n_in × n_out × w_max`).

### Tiga mekanisme, tiga skala berbeda

| Mekanisme | Yang dikontrol | BUKAN |
|---|---|---|
| **theta** (intrinsic) | **DAYA** — laju total populasi dibatasi | Kesamaan antar neuron |
| **column scaling** | **DRIVE per neuron** — disamakan antar neuron | Batas atas |
| **lateral inhibition** | **KOMPETISI** — siapa yang menang | Rate absolut |

Bukti bahwa theta BUKAN equalizer: pada input acak yang sudah seimbang,
CV laju justru **naik** dari 0.014 (LIF biasa) jadi 0.036 (Adaptive).
Theta adalah rem global, bukan penyamarataan.

Bukti bahwa inhibition BUKAN pembatas rate: tanpa homeostasis,
lateral inhibition sudah menyeimbangkan drive di ~331 — sistem reach
equilibrium sendiri. Yang gagal menahan adalah positive feedback
dari STDP dengan η asimetris.

## Temuan kedua: η asimetris itu wajib

Dengan `eta_plus == eta_minus`, STDP punya **fixed point** — LTP dan LTD
semenyakalkan sebelum bobot mencapai batas. Drive stabil di ~331, tidak
naik, tidak turun.

Konsekuensi yang berbahaya: kalau eksperimen homeostasis dilakukan dengan
η simetris, hasilnya akan menyimpulkan **"homeostasis tidak diperlukan"** —
dan kesimpulan itu salah.

Diehl-Cook memakai rasio ~100× (`nu_ee_post = 1e-2`, `nu_ee_pre = 1e-4`).
Di gate test kita pakai 20×, cukup untuk menunjukkan efeknya tanpa
langsung saturasi di epoch pertama.

## Keputusan

1. **Dokumentasikan pembagian kerja ini** di docstring ketiga modul dan di
   `docs/theory.md`. naskah yang belum dipisah.
2. **`theta_ss = beta_theta × rate × tau_theta`** adalah rumus tuning
   utama untuk Phase 3. Yang penting menjaga produknya kecil, bukan salah
   satu saja. Diuji di `test_matched_beta_times_tau_gives_matched_rate`.
3. **Setiap eksperimen homeostasis wajib memakai η asimetris.** Ditambahkan
   sebagai test (`test_equal_eta_stdp_is_self_limiting`) supaya tidak
   terulang.
4. **Mode τ_θ**: `TIME_SCALES` menyediakan `demo_seconds` (untuk test)
   sampai `diehl_cook_2.8h` (untuk Phase 3). Default adalah
   `realistic_hour`.

## Risiko yang terdeteksi

**Dead neuron.** Dengan `beta_theta` besar dan `tau_theta` panjang, θ menembus
jangkauan v yang bisa dicapai dan neuron berhenti permanen. Diuji di
`test_theta_can_kill_neurons_if_beta_too_big`.

Gejalanya: seluruh jaringan diam — dan karena itu akan salah dikira
"homeostasis terlalu kuat" padahal masalahnya justru sebaliknya:
homeostasis terlalu lemah untuk menjaga θ tetap di jangkauan.

Mitigasi Phase 3: pantau `max(θ) < v_th_max` sebagai metrik, dan clamp
θ ke batas atas jika perlu.

## Konsekuensi untuk Phase 3

- Inisialisasi bobot harus pakai `target_sum` yang sesuai (78 untuk 784 input).
- `eta_minus` harus jauh lebih kecil dari `eta_plus` — kalau tidak, tidak
  ada yang perlu distabilkan.
- Latar belakang (lateral inhibition) harus ikut diukur, karena dia punya
  efek sendiri terhadap stabilitas.