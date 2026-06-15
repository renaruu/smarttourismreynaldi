"""
=============================================================
  BAR CHART - KMEANS
  File sumber : results_wisata.csv + results_clustered.csv
=============================================================
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── 1. LOAD DATA ──────────────────────────────────────────
df_wisata = pd.read_csv("results_wisata.csv")
df_km     = pd.read_csv("results_clustered.csv")
df_wisata["cluster"] = df_km["cluster_kmeans"].values

# ── 2. STATISTIK ──────────────────────────────────────────
stats = df_wisata.groupby("cluster").agg(
    Jumlah  = ("name",    "count"),
    Rating  = ("rating",  "mean"),
    Ulasan  = ("reviews", "mean"),
).reset_index().sort_values("cluster")

WARNA = ["#2563EB","#16A34A","#DC2626","#D97706","#7C3AED"]
label_x = [f"Cluster {int(c)}" for c in stats["cluster"]]

print("=" * 50)
print("  STATISTIK KLASTER K-MEANS")
print("=" * 50)
for _, row in stats.iterrows():
    print(f"  Cluster {int(row['cluster'])} : {int(row['Jumlah'])} tempat | rating={row['Rating']:.2f} | ulasan={row['Ulasan']:,.0f}")

# ── 3. VISUALISASI ────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Distribusi Klaster K-Means (k=5)",
             fontsize=14, fontweight="bold", y=1.02)

# Chart 1: Jumlah
bars1 = axes[0].bar(label_x, stats["Jumlah"], color=WARNA,
                    edgecolor="white", linewidth=0.8, width=0.6)
axes[0].set_title("Jumlah Tempat Wisata\nBerdasarkan Klaster", fontsize=11, fontweight="bold")
axes[0].set_ylabel("Jumlah Tempat", fontsize=10)
axes[0].set_ylim(0, stats["Jumlah"].max() * 1.2)
axes[0].grid(axis="y", linestyle="--", alpha=0.4)
for bar, val in zip(bars1, stats["Jumlah"]):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 str(val), ha="center", va="bottom", fontsize=10, fontweight="bold")

# Chart 2: Rata-rata Rating
bars2 = axes[1].bar(label_x, stats["Rating"], color=WARNA,
                    edgecolor="white", linewidth=0.8, width=0.6)
axes[1].set_title("Rata-rata Rating\nBerdasarkan Klaster", fontsize=11, fontweight="bold")
axes[1].set_ylabel("Rata-rata Rating", fontsize=10)
axes[1].set_ylim(3.8, 5.2)
axes[1].grid(axis="y", linestyle="--", alpha=0.4)
for bar, val in zip(bars2, stats["Rating"]):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

# Chart 3: Rata-rata Ulasan
bars3 = axes[2].bar(label_x, stats["Ulasan"], color=WARNA,
                    edgecolor="white", linewidth=0.8, width=0.6)
axes[2].set_title("Rata-rata Ulasan\nBerdasarkan Klaster", fontsize=11, fontweight="bold")
axes[2].set_ylabel("Rata-rata Ulasan", fontsize=10)
axes[2].set_ylim(0, stats["Ulasan"].max() * 1.2)
axes[2].grid(axis="y", linestyle="--", alpha=0.4)
for bar, val in zip(bars3, stats["Ulasan"]):
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                 f"{val:,.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

patches = [mpatches.Patch(color=WARNA[i], label=f"Cluster {i}") for i in range(5)]
fig.legend(handles=patches, loc="lower center", ncol=5,
           fontsize=9, bbox_to_anchor=(0.5, -0.08), frameon=True)

plt.tight_layout()
plt.savefig("barchart_kmeans.png", dpi=150, bbox_inches="tight")
print("\n  ✅  Grafik tersimpan : barchart_kmeans.png")
plt.show()