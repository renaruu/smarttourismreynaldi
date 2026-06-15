"""
=============================================================
  DATA CLEANING - HAPUS MISSING VALUE
  File sumber : results_no_duplikat.csv
=============================================================

Kondisi missing value:
  name      :  0  (0.0%)  → tidak ada masalah
  rating    :  3  (0.4%)  → HAPUS baris
  reviews   : 15  (2.1%)  → HAPUS baris
  phone     : 340 (47.2%) → DIBIARKAN (wajar, banyak wisata tak punya telepon)
  website   : 443 (61.5%) → DIBIARKAN (wajar, banyak wisata tak punya website)

Strategi:
  ✅ rating  kosong → HAPUS baris
  ✅ reviews kosong → HAPUS baris
"""

import pandas as pd

# ── 1. LOAD DATA ──────────────────────────────────────────
FILE_IN  = "results.csv"
FILE_OUT = "results_misvalue.csv"

df = pd.read_csv(FILE_IN)

print("=" * 60)
print("  INFO DATASET AWAL")
print("=" * 60)
print(f"  Jumlah baris  : {len(df):,}")
print(f"  Jumlah kolom  : {df.shape[1]}")

# ── 2. DETEKSI MISSING VALUE ──────────────────────────────
print("\n" + "=" * 60)
print("  MISSING VALUE PER KOLOM")
print("=" * 60)
for col in df.columns:
    n   = df[col].isnull().sum()
    pct = n / len(df) * 100
    status = "⚠  akan dihapus" if col in ["rating", "reviews"] else \
             "✔  dibiarkan (wajar)" if n > 0 else "✔  lengkap"
    print(f"  {col:<12} : {n:3d} ({pct:5.1f}%)  {status}")

# ── 3. TAMPILKAN BARIS YANG AKAN DIHAPUS ──────────────────
print("\n  Baris yang akan dihapus (rating / reviews kosong):")
print("-" * 60)
masalah = df[df["rating"].isnull() | df["reviews"].isnull()] \
            [["name", "rating", "reviews", "category", "location"]]
print(masalah.to_string(index=False))

# ── 4. HAPUS BARIS MISSING VALUE ──────────────────────────
sebelum_rating  = df["rating"].isnull().sum()
sebelum_reviews = df["reviews"].isnull().sum()
sebelum_baris   = len(df)

df = df.dropna(subset=["rating", "reviews"]).reset_index(drop=True)

sesudah_rating  = df["rating"].isnull().sum()
sesudah_reviews = df["reviews"].isnull().sum()

print("\n" + "=" * 60)
print("  HASIL PENGHAPUSAN MISSING VALUE")
print("=" * 60)
print(f"  rating  : {sebelum_rating} missing → dihapus → sisa {sesudah_rating}")
print(f"  reviews : {sebelum_reviews} missing → dihapus → sisa {sesudah_reviews}")
print(f"\n  Baris dihapus : {sebelum_baris - len(df)}")
print(f"  Baris akhir   : {len(df):,}")

# ── 5. VERIFIKASI ─────────────────────────────────────────
total_missing = df[["rating", "reviews"]].isnull().sum().sum()
print(f"\n  Verifikasi missing (rating + reviews) : {total_missing} ✅")

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
print(df.head().to_string(index=False))