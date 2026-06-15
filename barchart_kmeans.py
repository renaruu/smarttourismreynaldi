"""
=============================================================
  VISUALISASI KATEGORI POPULARITAS CLUSTER
  File sumber : results_clustered.csv + results_wisata.csv
  Algoritma   : KMeans (Silhouette terbaik = 0.5529)
=============================================================
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── 1. LOAD DATA ──────────────────────────────────────────
df_clustered = pd.read_csv("results_clustered.csv")
df_wisata    = pd.read_csv("results_wisata.csv")

df_wisata["cluster"] = df_clustered["cluster_kmeans"]

# ── 2. DEFINISI LABEL & WARNA KATEGORI ───────────────────
KATEGORI = {
    0: ("Wisata Sangat Populer", "#DC2626"),
    1: ("Wisata Populer",        "#2563EB"),
    2: ("Wisata Standar", "#6B7280"),
    3: ("Wisata Potensial",      "#16A34A"),
    4: ("Wisata Cukup Populer",  "#D97706"),
}

df_wisata["kategori"] = df_wisata["cluster"].map(
    {k: v[0] for k, v in KATEGORI.items()}
)

# ── 3. HITUNG STATISTIK PER KATEGORI ─────────────────────
urutan = [KATEGORI[k][0] for k in [0, 3, 1, 4, 2]]
warna  = [KATEGORI[k][1] for k in [0, 3, 1, 4, 2]]

jumlah      = df_wisata.groupby("kategori").size().reindex(urutan)
avg_rating  = df_wisata.groupby("kategori")["rating"].mean().reindex(urutan)
avg_reviews = df_wisata.groupby("kategori")["reviews"].mean().reindex(urutan)

print("=" * 60)
print("  RINGKASAN KATEGORI POPULARITAS")
print("=" * 60)
for i, kat in enumerate(urutan):
    print(f"\n  {kat}")
    print(f"    Jumlah tempat    : {jumlah[kat]}")
    print(f"    Rata-rata rating : {avg_rating[kat]:.2f}")
    print(f"    Rata-rata ulasan : {avg_reviews[kat]:,.0f}")

# ── 4. VISUALISASI ────────────────────────────────────────
label_singkat = [
    "Wisata\nSangat Populer",
    "Wisata\nPotensial",
    "Wisata\nPopuler",
    "Wisata\nCukup Populer",
    "Wisata\nStandar",
]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Kategori Popularitas Tempat Wisata Jakarta\n(Berdasarkan KMeans Clustering, k=5)",
             fontsize=14, fontweight="bold", y=1.02)

# ── Chart 1: Jumlah Tempat ────────────────────────────────
bars1 = axes[0].bar(label_singkat, jumlah.values, color=warna,
                    edgecolor="white", linewidth=0.8, width=0.6)
axes[0].set_title("Jumlah Tempat Wisata\nBerdasarkan Klaster",
                  fontsize=11, fontweight="bold", pad=10)
axes[0].set_ylabel("Jumlah Tempat", fontsize=10)
axes[0].set_ylim(0, jumlah.max() * 1.2)
axes[0].grid(axis="y", linestyle="--", alpha=0.4)
axes[0].tick_params(axis="x", labelsize=8)

for bar, val in zip(bars1, jumlah.values):
    axes[0].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 3,
                 str(val), ha="center", va="bottom",
                 fontsize=10, fontweight="bold")

# ── Chart 2: Rata-rata Rating ─────────────────────────────
bars2 = axes[1].bar(label_singkat, avg_rating.values, color=warna,
                    edgecolor="white", linewidth=0.8, width=0.6)
axes[1].set_title("Rata-rata Rating\nBerdasarkan Klaster",
                  fontsize=11, fontweight="bold", pad=10)
axes[1].set_ylabel("Rata-rata Rating", fontsize=10)
axes[1].set_ylim(3.8, 5.1)
axes[1].grid(axis="y", linestyle="--", alpha=0.4)
axes[1].tick_params(axis="x", labelsize=8)

for bar, val in zip(bars2, avg_rating.values):
    axes[1].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.02,
                 f"{val:.2f}", ha="center", va="bottom",
                 fontsize=10, fontweight="bold")

# ── Chart 3: Rata-rata Ulasan ─────────────────────────────
bars3 = axes[2].bar(label_singkat, avg_reviews.values, color=warna,
                    edgecolor="white", linewidth=0.8, width=0.6)
axes[2].set_title("Rata-rata Jumlah Ulasan\nBerdasarkan Klaster",
                  fontsize=11, fontweight="bold", pad=10)
axes[2].set_ylabel("Rata-rata Ulasan", fontsize=10)
axes[2].set_ylim(0, avg_reviews.max() * 1.2)
axes[2].grid(axis="y", linestyle="--", alpha=0.4)
axes[2].tick_params(axis="x", labelsize=8)

for bar, val in zip(bars3, avg_reviews.values):
    axes[2].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 300,
                 f"{val:,.0f}", ha="center", va="bottom",
                 fontsize=9, fontweight="bold")

# ── Legend ────────────────────────────────────────────────
klaster_urutan = [0, 3, 1, 4, 2]
patches = [
    mpatches.Patch(color=KATEGORI[k][1], label=f"Cluster {k} — {KATEGORI[k][0]}")
    for k in klaster_urutan
]
fig.legend(handles=patches, loc="lower center", ncol=5,
           fontsize=9, bbox_to_anchor=(0.5, -0.08),
           frameon=True, edgecolor="#cccccc")

plt.tight_layout()
plt.savefig("visualisasi_kategori.png", dpi=150, bbox_inches="tight")
print("  Grafik tersimpan : visualisasi_kategori.png")
plt.show()