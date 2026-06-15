"""
=============================================================
  DATA CLEANING - HAPUS DUPLIKAT
  File sumber : results_cleaned.csv  (output seleksi atribut)
  Dibaca dari  : folder yang sama dengan script ini
=============================================================
"""

import pandas as pd

# ── 1. LOAD DATA ──────────────────────────────────────────
FILE_IN  = "results_misvalue.csv"
FILE_OUT = "results_no_duplikat.csv"

df = pd.read_csv(FILE_IN)

print("=" * 60)
print("  INFO DATASET AWAL")
print("=" * 60)
print(f"  Jumlah baris  : {len(df):,}")
print(f"  Jumlah kolom  : {df.shape[1]}")

# ── 2. DETEKSI DUPLIKAT ───────────────────────────────────
#
#  Tiga skenario pengecekan duplikat:
#
#  [A] Duplikat IDENTIK    → semua kolom sama persis
#  [B] Duplikat NAMA SAJA  → tempat sama, lokasi beda
#                            (muncul di banyak wilayah karena scraping)
#  [C] Duplikat NAMA+LOKASI→ tempat & kota sama, kategori beda

dup_semua     = df.duplicated().sum()
dup_nama      = df.duplicated(subset=["name"]).sum()
dup_koordinat = df.duplicated(subset=["latitude", "longitude"]).sum()

print("\n" + "=" * 60)
print("  HASIL DETEKSI DUPLIKAT")
print("=" * 60)
print(f"  [A] Duplikat identik (semua kolom)    : {dup_semua:,} baris")
print(f"  [B] Duplikat nama saja                : {dup_nama:,} baris")
print(f"  [C] Duplikat koordinat (lat+lon)      : {dup_koordinat:,} baris  ← yang dihapus")

# ── 3. TAMPILKAN CONTOH DUPLIKAT ──────────────────────────
print("\n  Contoh baris duplikat berdasarkan koordinat:")
print("-" * 60)
contoh = df[df.duplicated(subset=["latitude", "longitude"], keep=False)] \
           .sort_values("name") \
           .head(6)[["name", "rating", "location", "latitude", "longitude"]]
print(contoh.to_string(index=False))

# ── 4. HAPUS DUPLIKAT ─────────────────────────────────────
#
#  Strategi yang dipilih:
#  → Hapus duplikat berdasarkan "latitude" + "longitude"
#    karena tempat yang sama muncul di banyak baris dengan
#    kolom "location" berbeda (efek scraping per wilayah).
#    Koordinat adalah penanda unik lokasi fisik yang paling akurat.
#  → keep="first" → pertahankan kemunculan pertama

sebelum = len(df)

df_bersih = df.drop_duplicates(subset=["latitude", "longitude"], keep="first") \
              .reset_index(drop=True)

sesudah  = len(df_bersih)
terhapus = sebelum - sesudah

print("\n" + "=" * 60)
print("  HASIL PENGHAPUSAN DUPLIKAT")
print("=" * 60)
print(f"  Baris sebelum   : {sebelum:,}")
print(f"  Baris dihapus   : {terhapus:,}")
print(f"  Baris sesudah   : {sesudah:,}")
print(f"  Metode          : drop_duplicates(['latitude','longitude'], keep='first')")

# ── 5. VERIFIKASI HASIL ───────────────────────────────────
sisa_dup = df_bersih.duplicated(subset=["latitude", "longitude"]).sum()
print(f"\n  Verifikasi duplikat tersisa : {sisa_dup} baris ✅")

# ── 6. SIMPAN HASIL ───────────────────────────────────────
df_bersih.to_csv(FILE_OUT, index=False)

print("\n" + "=" * 60)
print(f"  ✅  File tersimpan : {FILE_OUT}")
print(f"  📋  Shape akhir    : {df_bersih.shape[0]} baris × {df_bersih.shape[1]} kolom")
print("=" * 60)

# ── 7. PRATINJAU ──────────────────────────────────────────
print("\n  Pratinjau 5 baris pertama data bersih:\n")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)
pd.set_option("display.max_colwidth", 30)
print(df_bersih.head().to_string(index=False))