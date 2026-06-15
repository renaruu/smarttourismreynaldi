"""
=============================================================
  FIX results_wisata.csv
  Jalankan script ini setelah edit manual di Excel
  sebelum menjalankan normalisasi.py
=============================================================

Masalah yang diperbaiki:
  1. Separator berubah dari koma (,) ke titik koma (;) akibat Excel
  2. Koordinat latitude & longitude kehilangan titik desimal
  3. Baris bermasalah akibat nama tempat mengandung koma
"""

import pandas as pd
import re

# ── 1. DETEKSI SEPARATOR ──────────────────────────────────
with open("results_wisata.csv", "r", encoding="utf-8") as f:
    baris_pertama = f.readline()

sep = ";" if ";" in baris_pertama else ","
print(f"  Separator terdeteksi : '{sep}'")

# ── 2. BACA FILE ──────────────────────────────────────────
df = pd.read_csv("results_wisata.csv", sep=sep, on_bad_lines="skip")
df = df[[c for c in df.columns if not c.startswith("Unnamed")]]

print(f"  Shape awal           : {df.shape}")

# ── 3. HAPUS BARIS BERMASALAH (koordinat tidak valid) ─────
def is_numeric(val):
    try:
        float(str(val).replace(".", "").replace(",", "").replace("-", ""))
        return True
    except:
        return False

mask_ok     = df["latitude"].apply(is_numeric)
baris_hapus = df[~mask_ok]
if len(baris_hapus) > 0:
    print(f"\n  Baris dihapus ({len(baris_hapus)}):")
    for _, row in baris_hapus.iterrows():
        print(f"    - {row['name']}")
df = df[mask_ok].reset_index(drop=True)

# ── 4. PERBAIKI KOORDINAT ─────────────────────────────────
def fix_coord(val, is_lat=True):
    try:
        clean = str(val).replace(".", "").replace(",", "")
        num   = float(clean)
        for divisor in [10000000, 1000000, 100000]:
            result = num / divisor
            if is_lat and -8 <= result <= -5:
                return round(result, 7)
            if not is_lat and 105 <= result <= 108:
                return round(result, 7)
        return round(num / 10000000, 7)
    except:
        return None

# Cek apakah koordinat perlu diperbaiki
lat_sample = df["latitude"].iloc[0]
perlu_fix  = abs(float(str(lat_sample).replace(".", "").replace(",", ""))) > 100

if perlu_fix:
    print("\n  Memperbaiki koordinat...")
    df["latitude"]  = df["latitude"].apply(lambda x: fix_coord(x, is_lat=True))
    df["longitude"] = df["longitude"].apply(lambda x: fix_coord(x, is_lat=False))
    df = df.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
else:
    print("\n  Koordinat sudah benar, tidak perlu diperbaiki ✅")

# ── 5. PERBAIKI TIPE DATA ─────────────────────────────────
df["rating"]  = pd.to_numeric(df["rating"],  errors="coerce")
df["reviews"] = pd.to_numeric(df["reviews"], errors="coerce").fillna(0).astype(int)

# ── 6. BERSIHKAN NAMA ─────────────────────────────────────
def bersihkan_nama(nama):
    if pd.isna(nama):
        return nama
    nama = re.sub(r'[\u0600-\u06FF\u0750-\u077F]+', '', nama)
    nama = nama.replace(',', '').replace(';', '')
    nama = ' '.join(nama.split())
    return nama.strip()

df["name"] = df["name"].apply(bersihkan_nama)

# ── 7. HAPUS DATA DI LUAR JAKARTA ─────────────────────────
sebelum = len(df)
df = df[(df["latitude"].between(-6.5, -6.0)) &
        (df["longitude"].between(106.6, 107.1))].reset_index(drop=True)
if sebelum - len(df) > 0:
    print(f"\n  Dihapus (di luar Jakarta) : {sebelum - len(df)} baris")

# ── 8. SIMPAN ─────────────────────────────────────────────
df.to_csv("results_wisata.csv", index=False)

print(f"\n  Shape akhir  : {df.shape}")
print(f"  Lat range    : {df['latitude'].min():.6f} to {df['latitude'].max():.6f}")
print(f"  Lon range    : {df['longitude'].min():.6f} to {df['longitude'].max():.6f}")
print(f"\n  ✅  results_wisata.csv berhasil diperbaiki!")
print(f"  Sekarang bisa jalankan normalisasi.py")