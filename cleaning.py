"""
=============================================================
  DATA CLEANING - SELEKSI ATRIBUT
  File sumber : results.csv
  Total baris : 1.649 | Total kolom awal : 15
=============================================================

Kolom asli:
  name, search_term, rating, reviews, phone, website,
  latitude, longitude, place_id, url, status, business_id,
  extracted_at, category, location

Strategi seleksi:
  DIPERTAHANKAN  → kolom yang bernilai analitis
  DIHAPUS        → kolom teknis / redundan (place_id, url,
                   status, business_id, search_term, extracted_at)
"""

import pandas as pd
import os

# ── 1. LOAD DATA ──────────────────────────────────────────
FILE_IN  = "results.csv"          # ganti path jika perlu (letakkan di folder yang sama)
FILE_OUT = "results_cleaned.csv"  # output tersimpan di folder yang sama

df_raw = pd.read_csv(FILE_IN)

print("=" * 55)
print("  INFO DATASET AWAL")
print("=" * 55)
print(f"  Jumlah baris  : {len(df_raw):,}")
print(f"  Jumlah kolom  : {df_raw.shape[1]}")
print(f"\n  Daftar kolom  :")
for i, col in enumerate(df_raw.columns, 1):
    null_pct = df_raw[col].isnull().mean() * 100
    print(f"    {i:2}. {col:<15}  |  missing: {null_pct:.1f}%")

# ── 2. SELEKSI ATRIBUT ────────────────────────────────────
#  Kolom yang DIHAPUS beserta alasannya:
#  (Informasi teknis scraping, id internal, url panjang, dll)

KOLOM_HAPUS = [
    "search_term",
    "place_id",
    "url",            # Sudah ditambahkan ke daftar hapus
    "status",
    "business_id",
    "extracted_at",
    "phone",
    "website",
]

KOLOM_SIMPAN = [col for col in df_raw.columns if col not in KOLOM_HAPUS]

df = df_raw[KOLOM_SIMPAN].copy()

print("\n" + "=" * 55)
print("  HASIL SELEKSI ATRIBUT")
print("=" * 55)
print(f"  Kolom dihapus    : {len(KOLOM_HAPUS)} kolom  → {KOLOM_HAPUS}")
print(f"  Kolom dipertahan : {len(KOLOM_SIMPAN)} kolom → {KOLOM_SIMPAN}")

# ── 3. RINGKASAN NILAI KOSONG SETELAH SELEKSI ─────────────
print("\n  Nilai kosong per kolom (setelah seleksi):")
missing = df.isnull().sum()
for col, n in missing.items():
    pct = n / len(df) * 100
    flag = "  ⚠" if pct > 30 else ""
    print(f"    {col:<12} : {n:4d} ({pct:5.1f}%){flag}")

# ── 4. RESET INDEX ────────────────────────────────────────
df = df.reset_index(drop=True)

# ── 5. SIMPAN HASIL ───────────────────────────────────────
df.to_csv(FILE_OUT, index=False)

print("\n" + "=" * 55)
print(f"  ✅  File tersimpan  : {FILE_OUT}")
print(f"  📋  Shape akhir     : {df.shape[0]} baris × {df.shape[1]} kolom")
print("=" * 55)

# ── 6. PRATINJAU 5 BARIS PERTAMA ──────────────────────────
print("\n  Pratinjau data bersih:\n")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)
print(df.head())