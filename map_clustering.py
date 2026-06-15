"""
=============================================================
  MAP CLUSTERING - 5 ALGORITMA
  File sumber : results_wisata.csv
               results_clustered.csv
               results_hdbscan_optics.csv
  Output      : map_clustering.png
  Library     : contextily (basemap peta Jakarta)
                pip install contextily
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import contextily as ctx
import warnings
warnings.filterwarnings("ignore")

# ── 1. LOAD DATA ──────────────────────────────────────────
df_wisata = pd.read_csv("results_wisata.csv")
df_km     = pd.read_csv("results_clustered.csv")
df_hdb    = pd.read_csv("results_hdbscan_optics.csv")
df_norm   = pd.read_csv("results_normalisasi.csv")

from sklearn.metrics import silhouette_score

def hitung_sil(X, labels):
    mask = labels != -1
    if len(set(labels[mask])) > 1:
        return silhouette_score(X[mask], labels[mask])
    return 0.0

X = df_norm[["rating", "reviews"]].values
sil_km  = hitung_sil(X, df_km["cluster_kmeans"].values)
sil_kmd = hitung_sil(X, df_km["cluster_kmedoid"].values)
sil_fcm = hitung_sil(X, df_km["cluster_fuzzy"].values)
sil_hdb = hitung_sil(X, df_hdb["cluster_hdbscan"].values)
sil_opt = hitung_sil(X, df_hdb["cluster_optics"].values)

df_wisata["cluster_kmeans"]  = df_km["cluster_kmeans"].values
df_wisata["cluster_kmedoid"] = df_km["cluster_kmedoid"].values
df_wisata["cluster_fuzzy"]   = df_km["cluster_fuzzy"].values
df_wisata["cluster_hdbscan"] = df_hdb["cluster_hdbscan"].values
df_wisata["cluster_optics"]  = df_hdb["cluster_optics"].values

# ── 2. KONVERSI KOORDINAT KE WEB MERCATOR ─────────────────
import pyproj

transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
lon = df_wisata["longitude"].values
lat = df_wisata["latitude"].values
x, y = transformer.transform(lon, lat)
df_wisata["x"] = x
df_wisata["y"] = y

# ── 3. WARNA ──────────────────────────────────────────────
WARNA = {
    -1: "#AAAAAA",
     0: "#2563EB",
     1: "#16A34A",
     2: "#DC2626",
     3: "#D97706",
     4: "#7C3AED",
     5: "#0891B2",
     6: "#BE185D",
}

ALGORITMA = [
    ("K-Means",        "cluster_kmeans",  sil_km),
    ("K-Medoids",       "cluster_kmedoid", sil_kmd),
    ("Fuzzy C-Means", "cluster_fuzzy",   sil_fcm),
    ("HDBSCAN",       "cluster_hdbscan", sil_hdb),
    ("OPTICS",        "cluster_optics",  sil_opt),
]

# ── 4. PLOT ───────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(20, 13))
fig.suptitle("Map Clustering Tempat Wisata Jakarta — 5 Algoritma",
             fontsize=15, fontweight="bold", y=1.01)

axes_flat = axes.flatten()

for idx, (nama_algo, col_cluster, sil) in enumerate(ALGORITMA):
    ax     = axes_flat[idx]
    labels = df_wisata[col_cluster].values

    for c in sorted(set(labels)):
        mask  = labels == c
        warna = WARNA.get(int(c), "#999999")
        label = "Noise" if c == -1 else f"Cluster {c}"
        alpha = 0.4 if c == -1 else 0.85

        ax.scatter(
            df_wisata.loc[mask, "x"],
            df_wisata.loc[mask, "y"],
            c=warna, s=18, alpha=alpha,
            edgecolors="white", linewidths=0.3,
            label=f"{label} ({mask.sum()})"
        )

    # Tambahkan basemap peta Jakarta
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom=12)

    ax.set_title(f"{nama_algo}\nSilhouette: {sil:.4f}",
                 fontsize=11, fontweight="bold", pad=8)
    ax.set_axis_off()
    ax.legend(fontsize=7, loc="upper right",
              framealpha=0.8, markerscale=1.2)

# Subplot terakhir: tabel silhouette score
axes_flat[5].axis("off")
axes_flat[5].text(0.5, 0.5,
    f"Silhouette Score\n\n"
    f"K-Means        : {sil_km:.4f}\n"
    f"K-Medoids      : {sil_kmd:.4f}\n"
    f"Fuzzy C-Means : {sil_fcm:.4f}\n"
    f"HDBSCAN       : {sil_hdb:.4f}\n"
    f"OPTICS        : {sil_opt:.4f}  \u2190 terbaik",
    transform=axes_flat[5].transAxes,
    ha="center", va="center", fontsize=12,
    bbox=dict(boxstyle="round,pad=0.8", facecolor="#F0F9FF",
              edgecolor="#2563EB", linewidth=1.5))

plt.tight_layout()
plt.savefig("map_clustering.png", dpi=150, bbox_inches="tight")
print("  ✅  Grafik tersimpan : map_clustering.png")
plt.show()