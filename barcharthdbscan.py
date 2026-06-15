"""
=============================================================
  BAR CHART - HDBSCAN
  File sumber : results_wisata.csv + results_hdbscan_optics.csv
=============================================================
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── 1. LOAD DATA ──────────────────────────────────────────
df_wisata = pd.read_csv("results_wisata.csv")
df_hdb    = pd.read_csv("results_hdbscan_optics.csv")
df_wisata["cluster"] = df_hdb["cluster_hdbscan"].values

# ── 2. STATISTIK ──────────────────────────────────────────
urutan  = sorted(df_wisata["cluster"].unique())
label_x = ["Noise" if int(c) == -1 else f"Cluster {int(c)}" for c in urutan]

WARNA = {
    0: "#2563EB",
    1: "#16A34A",
    2: "#DC2626",
    3: "#D97706",
    4: "#7C3AED",
    5: "#0891B2",
}

warna = [WARNA.get(int(c), "#999999") for c in urutan]

stats = df_wisata.groupby("cluster").agg(
    Jumlah  = ("name",    "count"),
    Rating  = ("rating",  "mean"),
    Ulasan  = ("reviews", "mean"),
).reindex(urutan)

# ── 3. VISUALISASI ────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(f"Distribusi Klaster HDBSCAN — {len(df_wisata)} Data",
             fontsize=14, fontweight="bold", y=1.02)

# Chart 1: Jumlah
bars1 = axes[0].bar(label_x, stats["Jumlah"], color=warna,
                    edgecolor="white", linewidth=0.8, width=0.6)
axes[0].set_title("Jumlah Tempat Wisata\nBerdasarkan Klaster", fontsize=11, fontweight="bold")
axes[0].set_ylabel("Jumlah Tempat", fontsize=10)
axes[0].set_ylim(0, stats["Jumlah"].max() * 1.2)
axes[0].grid(axis="y", linestyle="--", alpha=0.4)
axes[0].tick_params(axis="x", labelsize=9)
for bar, val in zip(bars1, stats["Jumlah"]):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 str(int(val)), ha="center", va="bottom", fontsize=10, fontweight="bold")

# Chart 2: Rata-rata Rating
bars2 = axes[1].bar(label_x, stats["Rating"], color=warna,
                    edgecolor="white", linewidth=0.8, width=0.6)
axes[1].set_title("Rata-rata Rating\nBerdasarkan Klaster", fontsize=11, fontweight="bold")
axes[1].set_ylabel("Rata-rata Rating", fontsize=10)
axes[1].set_ylim(3.8, 5.2)
axes[1].grid(axis="y", linestyle="--", alpha=0.4)
axes[1].tick_params(axis="x", labelsize=9)
for bar, val in zip(bars2, stats["Rating"]):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

# Chart 3: Rata-rata Ulasan
bars3 = axes[2].bar(label_x, stats["Ulasan"], color=warna,
                    edgecolor="white", linewidth=0.8, width=0.6)
axes[2].set_title("Rata-rata Ulasan\nBerdasarkan Klaster", fontsize=11, fontweight="bold")
axes[2].set_ylabel("Rata-rata Ulasan", fontsize=10)
axes[2].set_ylim(0, stats["Ulasan"].max() * 1.2)
axes[2].grid(axis="y", linestyle="--", alpha=0.4)
axes[2].tick_params(axis="x", labelsize=9)
for bar, val in zip(bars3, stats["Ulasan"]):
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                 f"{val:,.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

# ── Legend ────────────────────────────────────────────────
patches = [mpatches.Patch(color=WARNA.get(int(c), "#999999"),
           label="Noise" if int(c) == -1 else f"Cluster {int(c)}") for c in urutan]
fig.legend(handles=patches, loc="lower center", ncol=6,
           fontsize=9, bbox_to_anchor=(0.5, -0.08), frameon=True)

plt.tight_layout()
plt.savefig("barchart_hdbscan.png", dpi=150, bbox_inches="tight")
print("\n  ✅  Grafik tersimpan : barchart_hdbscan.png")
plt.show()