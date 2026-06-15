"""
=============================================================
  CLUSTERING HDBSCAN & OPTICS + EVALUASI SILHOUETTE SCORE
  File sumber : results_normalisasi.csv
  Algoritma   : HDBSCAN, OPTICS
  Fitur       : rating, reviews
  Library     : from sklearn.cluster import HDBSCAN, OPTICS
                (sklearn >= 1.3, tidak perlu install hdbscan terpisah)
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import HDBSCAN, OPTICS
from sklearn.metrics import silhouette_score

# ── 1. LOAD DATA ──────────────────────────────────────────
FILE_IN  = "results_normalisasi.csv"
FILE_OUT = "results_hdbscan_optics.csv"

df = pd.read_csv(FILE_IN)
X  = df[["rating", "reviews"]].values

print("=" * 60)
print("  CLUSTERING HDBSCAN & OPTICS")
print("=" * 60)
print(f"  Jumlah data : {len(X):,}")
print(f"  Fitur       : rating, reviews")

# ── 2. HDBSCAN ────────────────────────────────────────────
print("\n" + "=" * 60)
print("  [1] HDBSCAN")
print("=" * 60)
print("  Parameter : min_cluster_size=60, min_samples=5")

hdb       = HDBSCAN(min_cluster_size=60, min_samples=5, copy=True)
label_hdb = hdb.fit_predict(X)

n_cluster_hdb = len(set(label_hdb)) - (1 if -1 in label_hdb else 0)
n_noise_hdb   = (label_hdb == -1).sum()
mask_hdb      = label_hdb != -1
sil_hdb       = silhouette_score(X[mask_hdb], label_hdb[mask_hdb]) if n_cluster_hdb > 1 else 0

print(f"\n  Jumlah cluster terbentuk : {n_cluster_hdb}")
print(f"  Data noise (label -1)    : {n_noise_hdb} ({n_noise_hdb/len(X)*100:.1f}%)")
print(f"  Silhouette Score         : {sil_hdb:.4f} (dari {mask_hdb.sum()} data non-noise)")
print(f"\n  Distribusi cluster:")
for c, n in sorted(pd.Series(label_hdb).value_counts().items()):
    label = "Noise" if c == -1 else f"Cluster {c}"
    print(f"    {label:<12} : {n} data")

df["cluster_hdbscan"] = label_hdb

# ── 3. OPTICS ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  [2] OPTICS")
print("=" * 60)
print("  Parameter : min_samples=20, xi=0.05")

opt       = OPTICS(min_samples=20, xi=0.05, min_cluster_size=0.05)
label_opt = opt.fit_predict(X)

n_cluster_opt = len(set(label_opt)) - (1 if -1 in label_opt else 0)
n_noise_opt   = (label_opt == -1).sum()
mask_opt      = label_opt != -1
sil_opt       = silhouette_score(X[mask_opt], label_opt[mask_opt]) if n_cluster_opt > 1 else 0

print(f"\n  Jumlah cluster terbentuk : {n_cluster_opt}")
print(f"  Data noise (label -1)    : {n_noise_opt} ({n_noise_opt/len(X)*100:.1f}%)")
print(f"  Silhouette Score         : {sil_opt:.4f} (dari {mask_opt.sum()} data non-noise)")
print(f"\n  Distribusi cluster:")
for c, n in sorted(pd.Series(label_opt).value_counts().items()):
    label = "Noise" if c == -1 else f"Cluster {c}"
    print(f"    {label:<12} : {n} data")

df["cluster_optics"] = label_opt

# ── 4. PERBANDINGAN SILHOUETTE SCORE ──────────────────────
print("\n" + "=" * 60)
print("  PERBANDINGAN SILHOUETTE SCORE")
print("=" * 60)

hasil   = {"HDBSCAN": sil_hdb, "OPTICS": sil_opt}
terbaik = max(hasil, key=hasil.get)

for nama, skor in hasil.items():
    flag = " <- terbaik" if nama == terbaik else ""
    print(f"  {nama:<10} : {skor:.4f}{flag}")

print(f"\n  Interpretasi Silhouette Score:")
print(f"    0.71 - 1.00  : Struktur cluster sangat kuat")
print(f"    0.51 - 0.70  : Struktur cluster cukup kuat")
print(f"    0.26 - 0.50  : Struktur cluster lemah")
print(f"    < 0.25       : Tidak ada struktur yang jelas")

# ── 5. SIMPAN CSV SEBELUM TAMPILKAN GRAFIK ────────────────
df.to_csv(FILE_OUT, index=False)
print(f"\n  Dataset tersimpan : {FILE_OUT}")
print(f"  Kolom baru        : cluster_hdbscan, cluster_optics")

# ── 6. VISUALISASI ────────────────────────────────────────
WARNA_MAP = {
    -1: "#CCCCCC",
     0: "#2563EB",
     1: "#16A34A",
     2: "#DC2626",
     3: "#D97706",
     4: "#7C3AED",
     5: "#0891B2",
     6: "#BE185D",
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Hasil Clustering — HDBSCAN vs OPTICS",
             fontsize=14, fontweight="bold", y=1.02)

def plot_cluster(ax, labels, title, skor, n_cluster, n_noise):
    for c in sorted(set(labels)):
        mask  = labels == c
        warna = WARNA_MAP.get(c, "#999999")
        nama  = "Noise" if c == -1 else f"Cluster {c}"
        ax.scatter(X[mask, 0], X[mask, 1],
                   c=warna, label=f"{nama} ({mask.sum()})",
                   alpha=0.4 if c == -1 else 0.7, s=25)
    ax.set_title(f"{title}\nSilhouette: {skor:.4f} | {n_cluster} cluster | {n_noise} noise",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Rating (norm)", fontsize=10)
    ax.set_ylabel("Reviews (norm)", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.4)

plot_cluster(axes[0], label_hdb, "HDBSCAN", sil_hdb, n_cluster_hdb, n_noise_hdb)
plot_cluster(axes[1], label_opt, "OPTICS",  sil_opt, n_cluster_opt, n_noise_opt)

plt.tight_layout()
plt.savefig("hdbscan_optics_hasil.png", dpi=150, bbox_inches="tight")
print(f"  Grafik tersimpan  : hdbscan_optics_hasil.png")
plt.show()