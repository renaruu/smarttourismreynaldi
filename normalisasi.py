"""
=============================================================
  DATA CLEANING - NORMALISASI MIN-MAX
  File sumber : results_wisata.csv
=============================================================

Kolom yang dinormalisasi:
  - rating   (skala asli: 1.0 – 5.0)
  - reviews  (skala asli: 10 – 173.725)

Kolom yang TIDAK dinormalisasi:
  - latitude, longitude  → dipakai untuk visualisasi peta
  - name, phone, website, category, location → data non-numerik

Rumus Min-Max:
  X_norm = (X - X_min) / (X_max - X_min)
  Hasil   : nilai antara 0.0 – 1.0
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ── 1. LOAD DATA ──────────────────────────────────────────
FILE_IN  = "results_wisata.csv"
FILE_OUT = "results_normalisasi.csv"

df = pd.read_csv(FILE_IN)

print("=" * 60)
print("  INFO DATASET AWAL")
print("=" * 60)
print(f"  Jumlah baris  : {len(df):,}")
print(f"  Jumlah kolom  : {df.shape[1]}")

# ── 2. CEK STATISTIK SEBELUM NORMALISASI ──────────────────
KOLOM_NORMALISASI = ["rating", "reviews"]

print("\n" + "=" * 60)
print("  STATISTIK SEBELUM NORMALISASI")
print("=" * 60)
print(df[KOLOM_NORMALISASI].describe().round(4).to_string())

# ── 3. TERAPKAN MIN-MAX SCALER ────────────────────────────
scaler = MinMaxScaler()

df[KOLOM_NORMALISASI] = scaler.fit_transform(df[KOLOM_NORMALISASI])

# ── 4. CEK STATISTIK SETELAH NORMALISASI ──────────────────
print("\n" + "=" * 60)
print("  STATISTIK SETELAH NORMALISASI (skala 0.0 – 1.0)")
print("=" * 60)
print(df[KOLOM_NORMALISASI].describe().round(4).to_string())

print("\n  Nilai min & max per kolom setelah normalisasi:")
print("-" * 40)
for col in KOLOM_NORMALISASI:
    print(f"  {col:<10} : min = {df[col].min():.1f}  |  max = {df[col].max():.1f}")

# ── 5. KONFIRMASI KOLOM TIDAK DIUBAH ──────────────────────
print("\n" + "=" * 60)
print("  KOLOM TIDAK DINORMALISASI (tetap seperti semula)")
print("=" * 60)
kolom_tetap = ["latitude", "longitude"]
print(df[kolom_tetap].describe().round(6).to_string())

# ── 6. SIMPAN HASIL ───────────────────────────────────────
df.to_csv(FILE_OUT, index=False)

print("\n" + "=" * 60)
print(f"  ✅  File tersimpan : {FILE_OUT}")
print(f"  📋  Shape akhir    : {df.shape[0]} baris × {df.shape[1]} kolom")
print("=" * 60)

# ── 7. PRATINJAU ──────────────────────────────────────────
print("\n  Pratinjau 5 baris pertama:\n")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)
pd.set_option("display.max_colwidth", 30)
print(df[["name", "rating", "reviews", "latitude", "longitude", "category"]].head().to_string(index=False))