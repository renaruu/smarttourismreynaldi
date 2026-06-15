"""
=============================================================
  DATA CLEANING - SELEKSI TEMPAT WISATA
  File sumber : results_misvalue.csv
=============================================================

Strategi seleksi:
  1. Hapus baris dengan kata kunci bukan wisata di kolom name
     (RPTRA, warung, bubur, bakmi, dll)
  2. Hapus baris dengan reviews < 10
  3. Bersihkan karakter bermasalah di kolom name
     (koma, titik koma, karakter Arab/non-latin)
"""

import pandas as pd
import re

# ── 1. LOAD DATA ──────────────────────────────────────────
FILE_IN  = "results_misvalue.csv"
FILE_OUT = "results_wisata.csv"

df = pd.read_csv(FILE_IN)

print("=" * 60)
print("  INFO DATASET AWAL")
print("=" * 60)
print(f"  Jumlah baris  : {len(df):,}")
print(f"  Jumlah kolom  : {df.shape[1]}")
print(f"\n  Distribusi kategori:")
for cat, n in df["category"].value_counts().items():
    print(f"    {cat:<35} : {n}")

# ── 2. FILTER KATA KUNCI BUKAN WISATA ─────────────────────
KEYWORDS_HAPUS = [
    "RPTRA",
    "warung",
    "bubur",
    "bakmi", "bakmie",
    r"\bmie\b",
    r"\bnasi\b",
    "ayam goreng",
    r"\bsoto\b",
    "angkringan",
    "kopi es",
]

pattern = "|".join(KEYWORDS_HAPUS)
mask_kw  = df["name"].str.contains(pattern, case=False, na=False, regex=True)

print("\n" + "=" * 60)
print("  FILTER 1 — KATA KUNCI BUKAN WISATA")
print("=" * 60)
print(f"  Baris yang akan dihapus : {mask_kw.sum()}")
print("\n  Daftar baris terhapus:")
print("-" * 60)
terhapus_kw = df[mask_kw][["name", "category", "reviews"]].reset_index(drop=True)
print(terhapus_kw.to_string(index=False))

df_step1 = df[~mask_kw].reset_index(drop=True)

# ── 3. FILTER MINIMUM REVIEWS ─────────────────────────────
MIN_REVIEWS = 10

mask_rev = df_step1["reviews"] < MIN_REVIEWS

print("\n" + "=" * 60)
print(f"  FILTER 2 — REVIEWS < {MIN_REVIEWS}")
print("=" * 60)
print(f"  Baris yang akan dihapus : {mask_rev.sum()}")
print("\n  Daftar baris terhapus:")
print("-" * 60)
terhapus_rev = df_step1[mask_rev][["name", "category", "reviews"]].reset_index(drop=True)
print(terhapus_rev.to_string(index=False))

df_step2 = df_step1[~mask_rev].reset_index(drop=True)

# ── 4. RINGKASAN HASIL ────────────────────────────────────
print("\n" + "=" * 60)
print("  RINGKASAN HASIL SELEKSI")
print("=" * 60)
print(f"  Baris awal              : {len(df):,}")
print(f"  Dihapus (kata kunci)    : {mask_kw.sum()}")
print(f"  Dihapus (reviews < {MIN_REVIEWS}) : {mask_rev.sum()}")
print(f"  Baris akhir             : {len(df_step2):,}")

print(f"\n  Distribusi kategori setelah seleksi:")
for cat, n in df_step2["category"].value_counts().items():
    sebelum = df["category"].value_counts().get(cat, 0)
    print(f"    {cat:<35} : {n:3d}  (sebelum: {sebelum})")

# ── 5. VERIFIKASI ─────────────────────────────────────────
sisa_kw  = df_step2["name"].str.contains(pattern, case=False, na=False, regex=True).sum()
sisa_rev = (df_step2["reviews"] < MIN_REVIEWS).sum()
print(f"\n  Verifikasi kata kunci tersisa : {sisa_kw} ✅")
print(f"  Verifikasi reviews < {MIN_REVIEWS} tersisa : {sisa_rev} ✅")

# ── 6. BERSIHKAN KARAKTER BERMASALAH ──────────────────────
print("\n" + "=" * 60)
print("  BERSIHKAN KARAKTER BERMASALAH DI KOLOM NAME")
print("=" * 60)

def bersihkan_nama(nama):
    if pd.isna(nama):
        return nama
    # Hapus karakter Arab dan non-latin
    nama = re.sub(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+', '', nama)
    # Hapus koma dan titik koma (penyebab bug CSV)
    nama = nama.replace(',', '').replace(';', '')
    # Hapus spasi berlebih
    nama = ' '.join(nama.split())
    return nama.strip()

nama_sebelum = df_step2["name"].copy()
df_step2["name"] = df_step2["name"].apply(bersihkan_nama)

# Tampilkan nama yang berubah
berubah = df_step2[df_step2["name"] != nama_sebelum][["name"]]
if len(berubah) > 0:
    print(f"  Nama yang dibersihkan : {len(berubah)} baris")
    for i, row in berubah.iterrows():
        print(f"    Sebelum : {nama_sebelum[i]}")
        print(f"    Sesudah : {row['name']}")
        print()
else:
    print("  Tidak ada nama yang perlu dibersihkan ✅")

# ── 7. SIMPAN ─────────────────────────────────────────────
df_step2["reviews"] = df_step2["reviews"].astype(int)
df_step2.to_csv(FILE_OUT, index=False)

print("\n" + "=" * 60)
print(f"  ✅  File tersimpan : {FILE_OUT}")
print(f"  📋  Shape akhir    : {df_step2.shape[0]} baris × {df_step2.shape[1]} kolom")
print("=" * 60)

# ── 8. PRATINJAU ──────────────────────────────────────────
print("\n  Pratinjau 5 baris pertama:\n")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)
pd.set_option("display.max_colwidth", 35)
print(df_step2.head().to_string(index=False))