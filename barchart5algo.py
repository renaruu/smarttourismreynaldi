"""
=============================================================
  BAR CHART - 5 ALGORITMA CLUSTERING
  File sumber : results_wisata.csv
               results_clustered.csv
               results_hdbscan_optics.csv
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# ── 1. LOAD DATA ──────────────────────────────────────────
df_wisata = pd.read_csv("results_wisata.csv")
df_km     = pd.read_csv("results_clustered.csv")
df_hdb    = pd.read_csv("results_hdbscan_optics.csv")
df_norm   = pd.read_csv("results_normalisasi.csv")

X = df_norm[["rating", "reviews"]].values

df_wisata["cluster_kmeans"]  = df_km["cluster_kmeans"].values
df_wisata["cluster_kmedoid"] = df_km["cluster_kmedoid"].values
df_wisata["cluster_fuzzy"]   = df_km["cluster_fuzzy"].values
df_wisata["cluster_hdbscan"] = df_hdb["cluster_hdbscan"].values
df_wisata["cluster_optics"]  = df_hdb["cluster_optics"].values

# ── 2. HITUNG SILHOUETTE SCORE OTOMATIS ───────────────────
def hitung_sil(X, labels):
    mask = labels != -1
    if len(set(labels[mask])) > 1:
        return silhouette_score(X[mask], labels[mask])
    return 0.0

sil_km  = hitung_sil(X, df_wisata["cluster_kmeans"].values)
sil_kmd = hitung_sil(X, df_wisata["cluster_kmedoid"].values)
sil_fcm = hitung_sil(X, df_wisata["cluster_fuzzy"].values)
sil_hdb = hitung_sil(X, df_wisata["cluster_hdbscan"].values)
sil_opt = hitung_sil(X, df_wisata["cluster_optics"].values)

print("=" * 45)
print("  SILHOUETTE SCORE (otomatis)")
print("=" * 45)
print(f"  K-Means        : {sil_km:.4f}")
print(f"  K-Medoids      : {sil_kmd:.4f}")
print(f"  Fuzzy C-Means : {sil_fcm:.4f}")
print(f"  HDBSCAN       : {sil_hdb:.4f}")
print(f"  OPTICS        : {sil_opt:.4f}")

# ── 3. WARNA ──────────────────────────────────────────────
WARNA_CLUSTER = ["#2563EB","#16A34A","#DC2626","#D97706","#7C3AED","#0891B2","#BE185D","#84CC16"]
WARNA_NOISE   = "#CCCCCC"

ALGORITMA = {
    "K-Means"       : ("cluster_kmeans",  sil_km),
    "K-Medoids"      : ("cluster_kmedoid", sil_kmd),
    "Fuzzy C-Means": ("cluster_fuzzy",   sil_fcm),
    "HDBSCAN"      : ("cluster_hdbscan", sil_hdb),
    "OPTICS"       : ("cluster_optics",  sil_opt),
}

# ── 4. VISUALISASI ────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(20, 11))
fig.suptitle("Distribusi Klaster — 5 Algoritma Clustering",
             fontsize=15, fontweight="bold", y=1.01)

axes_flat = axes.flatten()

for idx, (nama_algo, (col, sil)) in enumerate(ALGORITMA.items()):
    ax     = axes_flat[idx]
    counts = df_wisata.groupby(col).size().sort_index()

    labels = []
    warna  = []
    for c in counts.index:
        if c == -1:
            labels.append("Noise")
            warna.append(WARNA_NOISE)
        else:
            labels.append(f"Cluster {c}")
            warna.append(WARNA_CLUSTER[int(c) % len(WARNA_CLUSTER)])

    bars = ax.bar(labels, counts.values, color=warna,
                  edgecolor="white", linewidth=0.8, width=0.6)

    ax.set_title(f"{nama_algo}\nSilhouette: {sil:.4f}",
                 fontsize=12, fontweight="bold", pad=8)
    ax.set_ylabel("Jumlah Tempat Wisata", fontsize=9)
    ax.set_ylim(0, counts.max() * 1.2)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.tick_params(axis="x", labelsize=9)

    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 2,
                str(val), ha="center", va="bottom",
                fontsize=9, fontweight="bold")

# Subplot terakhir: perbandingan silhouette score
ax_sil = axes_flat[5]
algo_names  = ["K-Means", "K-Medoids", "Fuzzy\nC-Means", "HDBSCAN", "OPTICS"]
sil_scores  = [sil_km, sil_kmd, sil_fcm, sil_hdb, sil_opt]
warna_bars  = ["#2563EB","#16A34A","#DC2626","#D97706","#7C3AED"]
terbaik_idx = sil_scores.index(max(sil_scores))

bars_sil = ax_sil.bar(algo_names, sil_scores,
                       color=warna_bars, edgecolor="white",
                       linewidth=0.8, width=0.6)

# Tandai terbaik
bars_sil[terbaik_idx].set_edgecolor("#DC2626")
bars_sil[terbaik_idx].set_linewidth(2.5)

ax_sil.set_title("Perbandingan Silhouette Score",
                 fontsize=12, fontweight="bold", pad=8)
ax_sil.set_ylabel("Silhouette Score", fontsize=9)
ax_sil.set_ylim(0, 1.0)
ax_sil.grid(axis="y", linestyle="--", alpha=0.4)
ax_sil.tick_params(axis="x", labelsize=9)

for i, (bar, val) in enumerate(zip(bars_sil, sil_scores)):
    ket = " ← terbaik" if i == terbaik_idx else ""
    ax_sil.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.02,
                f"{val:.4f}{ket}", ha="center", va="bottom",
                fontsize=8, fontweight="bold",
                color="#DC2626" if i == terbaik_idx else "black")

plt.tight_layout()
plt.savefig("barchart_5algo.png", dpi=150, bbox_inches="tight")
print("\n  ✅  Grafik tersimpan : barchart_5algo.png")
plt.show()