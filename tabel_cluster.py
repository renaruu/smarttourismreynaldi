"""
=============================================================
  TABEL HASIL KLASTERISASI - OPTICS
  File sumber : results_hdbscan_optics.csv + results_wisata.csv
  Algoritma   : OPTICS (Silhouette terbaik = 0.7791)
=============================================================
"""

import pandas as pd

# ── 1. LOAD DATA ──────────────────────────────────────────
df      = pd.read_csv("results_hdbscan_optics.csv")
df_asli = pd.read_csv("results_wisata.csv")

df["rating"]  = df_asli["rating"]
df["reviews"] = df_asli["reviews"].astype(int)

# ── 2. MAPPING LABEL ──────────────────────────────────────
LABEL = {
    7: "Wisata Sangat Populer",
    1: "Wisata Populer",
    0: "Wisata Cukup Populer",
    4: "Wisata Ramai",
    2: "Wisata Berpotensi",
    3: "Wisata Potensial",
    5: "Wisata Standar",
    6: "Wisata Cukup Dikenal",
}

df["kategori_optics"] = df["cluster_optics"].map(LABEL)

# ── 3. SUSUN TABEL FINAL ──────────────────────────────────
tabel = df[[
    "name", "rating", "reviews", "category", "location",
    "url", "latitude", "longitude",
    "cluster_optics", "kategori_optics"
]].copy()

tabel.columns = [
    "Nama Tempat", "Rating", "Ulasan", "Kategori Wisata", "Lokasi",
    "URL", "latitude", "longitude",
    "No Klaster", "Klaster OPTICS"
]

tabel = tabel.sort_values("Ulasan", ascending=False).reset_index(drop=True)
tabel.index += 1

# ── 4. TAMPILKAN DI TERMINAL ──────────────────────────────
print("=" * 65)
print("  TABEL HASIL KLASTERISASI OPTICS")
print("=" * 65)
print(f"  Total data : {len(tabel)}")

print("\n  Distribusi per klaster:")
print("-" * 45)
dist = tabel["Klaster OPTICS"].value_counts()
for kat, n in dist.items():
    pct = n / len(tabel) * 100
    print(f"  {kat:<25} : {n:3d} ({pct:.1f}%)")

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 15)
pd.set_option("display.width", 150)
pd.set_option("display.max_colwidth", 35)

print("\n  Pratinjau 10 data teratas (ulasan terbanyak):\n")
print(tabel.head(10).to_string())

# ── 6. SIMPAN CSV ─────────────────────────────────────────
tabel.to_csv("tabel_klaster_optics.csv", index=True, index_label="No")
print(f"\n  ✅  Tersimpan : tabel_klaster_optics.csv")
print(f"      Kolom     : Nama Tempat, Rating, Ulasan, Kategori Wisata,")
print(f"                  Lokasi, URL, latitude, longitude, No Klaster, Klaster OPTICS")

# ── 7. SIMPAN EXCEL ───────────────────────────────────────
try:
    import openpyxl

    urutan_klaster = [
        "Wisata Sangat Populer",
        "Wisata Populer",
        "Wisata Cukup Populer",
        "Wisata Ramai",
        "Wisata Potensial",
        "Wisata Tersembunyi",
        "Wisata Biasa",
        "Wisata kurang Dikenal",
    ]

    with pd.ExcelWriter("tabel_klaster_optics.xlsx", engine="openpyxl") as writer:

        # Sheet 1 — semua data
        tabel.to_excel(writer, sheet_name="Semua Data", index=True, index_label="No")

        # Sheet 2 — ringkasan per klaster
        ringkasan = tabel.groupby("Klaster OPTICS").agg(
            Jumlah_Tempat = ("Nama Tempat", "count"),
            Avg_Rating    = ("Rating",      "mean"),
            Avg_Ulasan    = ("Ulasan",      "mean"),
            Max_Ulasan    = ("Ulasan",      "max"),
        ).round(2)
        ringkasan = ringkasan.reindex([k for k in urutan_klaster if k in ringkasan.index])
        ringkasan.to_excel(writer, sheet_name="Ringkasan per Klaster")

        # Sheet per klaster
        for kat in urutan_klaster:
            subset = tabel[tabel["Klaster OPTICS"] == kat].copy()
            if len(subset) > 0:
                nama_sheet = kat[:25]
                subset.to_excel(writer, sheet_name=nama_sheet, index=True, index_label="No")

    print(f"  ✅  Tersimpan : tabel_klaster_optics.xlsx")
    print(f"      Sheet     : Semua Data + Ringkasan + 1 sheet per klaster")

except ImportError:
    print("  ⚠   openpyxl belum terinstall → pip install openpyxl")

print(f"\n  Total data tersimpan : {len(tabel)} baris ✅")