"""
=============================================================
  SCATTER PLOT - 5 ALGORITMA CLUSTERING
  File sumber : results_normalisasi.csv
               results_clustered.csv
               results_hdbscan_optics.csv
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# ── 1. LOAD DATA ──────────────────────────────────────────
df_norm = pd.read_csv("results_normalisasi.csv")
df_km   = pd.read_csv("results_clustered.csv")
df_hdb  = pd.read_csv("results_hdbscan_optics.csv")

X = df_norm[["rating", "reviews"]].values

label_kmeans  = df_km["cluster_kmeans"].values
label_kmedoid = df_km["cluster_kmedoid"].values
label_fuzzy   = df_km["cluster_fuzzy"].values
label_hdbscan = df_hdb["cluster_hdbscan"].values
label_optics  = df_hdb["cluster_optics"].values

# ── 2. HITUNG SILHOUETTE SCORE OTOMATIS ───────────────────
def hitung_sil(X, labels):
    mask = labels != -1
    if len(set(labels[mask])) > 1:
        return silhouette_score(X[mask], labels[mask])
    return 0.0

sil_km  = hitung_sil(X, label_kmeans)
sil_kmd = hitung_sil(X, label_kmedoid)
sil_fcm = hitung_sil(X, label_fuzzy)
sil_hdb = hitung_sil(X, label_hdbscan)
sil_opt = hitung_sil(X, label_optics)

terbaik = max(
    [("KMeans", sil_km), ("KMedoid", sil_kmd),
     ("Fuzzy C-Means", sil_fcm), ("HDBSCAN", sil_hdb), ("OPTICS", sil_opt)],
    key=lambda x: x[1]
)[0]

print("=" * 45)
print("  SILHOUETTE SCORE")
print("=" * 45)
print(f"  KMeans        : {sil_km:.4f}")
print(f"  KMedoid       : {sil_kmd:.4f}")
print(f"  Fuzzy C-Means : {sil_fcm:.4f}")
print(f"  HDBSCAN       : {sil_hdb:.4f}")
print(f"  OPTICS        : {sil_opt:.4f}  <- terbaik" if terbaik == "OPTICS" else f"  OPTICS        : {sil_opt:.4f}")
print(f"  Terbaik       : {terbaik}")

# ── 3. WARNA ──────────────────────────────────────────────
WARNA = {
    -1: "#CCCCCC",
     0: "#2563EB",
     1: "#16A34A",
     2: "#DC2626",
     3: "#D97706",
     4: "#7C3AED",
     5: "#0891B2",
     6: "#BE185D",
     7: "#84CC16",
}

# ── 4. PLOT ───────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Scatter Plot Hasil Clustering — 5 Algoritma",
             fontsize=15, fontweight="bold", y=1.01)

def plot_scatter(ax, labels, title, X):
    for c in sorted(set(labels)):
        mask  = labels == c
        warna = WARNA.get(int(c), "#999999")
        nama  = "Noise" if c == -1 else f"Cluster {c}"
        ax.scatter(X[mask, 0], X[mask, 1],
                   c=warna, label=f"{nama} ({mask.sum()})",
                   alpha=0.5 if c == -1 else 0.75,
                   s=30, edgecolors="white", linewidths=0.3)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel("Rating (norm)", fontsize=10)
    ax.set_ylabel("Reviews (norm)", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.4)

plot_scatter(axes[0, 0], label_kmeans,  f"KMeans (k=5)\nSilhouette: {sil_km:.4f}",        X)
plot_scatter(axes[0, 1], label_kmedoid, f"KMedoid (k=5)\nSilhouette: {sil_kmd:.4f}",      X)
plot_scatter(axes[0, 2], label_fuzzy,   f"Fuzzy C-Means (k=5)\nSilhouette: {sil_fcm:.4f}",X)
plot_scatter(axes[1, 0], label_hdbscan, f"HDBSCAN\nSilhouette: {sil_hdb:.4f}",            X)
plot_scatter(axes[1, 1], label_optics,  f"OPTICS\nSilhouette: {sil_opt:.4f}",             X)

# Kotak silhouette score
axes[1, 2].axis("off")
axes[1, 2].text(0.5, 0.5,
    f"Silhouette Score\n\n"
    f"KMeans        : {sil_km:.4f}\n"
    f"KMedoid       : {sil_kmd:.4f}\n"
    f"Fuzzy C-Means : {sil_fcm:.4f}\n"
    f"HDBSCAN       : {sil_hdb:.4f}\n"
    f"OPTICS        : {sil_opt:.4f}  \u2190 terbaik",
    transform=axes[1, 2].transAxes,
    ha="center", va="center", fontsize=11,
    bbox=dict(boxstyle="round,pad=0.8", facecolor="#F0F9FF",
              edgecolor="#2563EB", linewidth=1.5))

plt.tight_layout()
plt.savefig("scatter_plot.png", dpi=150, bbox_inches="tight")
print("\n  ✅  Grafik tersimpan : scatter_plot.png")
plt.show()