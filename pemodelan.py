"""
=============================================================
  CLUSTERING + EVALUASI SILHOUETTE SCORE
  File sumber : results_normalisasi.csv
  Algoritma   : KMeans, KMedoid, Fuzzy C-Means
  k           : 5 (hasil Elbow Method)
  Fitur       : rating, reviews
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, pairwise_distances
import kmedoids
import skfuzzy as fuzz

# ── 1. LOAD DATA ──────────────────────────────────────────
FILE_IN = "results_normalisasi.csv"

df = pd.read_csv(FILE_IN)
X  = df[["rating", "reviews"]].values
K  = 5

print("=" * 60)
print("  CLUSTERING + EVALUASI SILHOUETTE SCORE")
print("=" * 60)
print(f"  Jumlah data : {len(X):,}")
print(f"  Fitur       : rating, reviews")
print(f"  k           : {K}")

# ── 2. KMEANS ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  [1] K-MEANS")
print("=" * 60)

kmeans   = KMeans(n_clusters=K, init="k-means++", n_init=10, random_state=42)
label_km = kmeans.fit_predict(X)
sil_km   = silhouette_score(X, label_km)
df["cluster_kmeans"] = label_km

print(f"  Silhouette Score : {sil_km:.4f}")
print(f"  Distribusi cluster:")
for c, n in sorted(pd.Series(label_km).value_counts().items()):
    print(f"    Cluster {c} : {n} data")

# ── 3. K-MEDOIDS ────────────────────────────────────────────
print("\n" + "=" * 60)
print("  [2] K-MEDOIDS")
print("=" * 60)

D           = pairwise_distances(X, metric="euclidean")
km_result   = kmedoids.fasterpam(D, K, random_state=42)
label_kmd   = np.array(km_result.labels)
centers_kmd = X[km_result.medoids]
sil_kmd     = silhouette_score(X, label_kmd)
df["cluster_kmedoid"] = label_kmd

print(f"  Silhouette Score : {sil_kmd:.4f}")
print(f"  Distribusi cluster:")
for c, n in sorted(pd.Series(label_kmd).value_counts().items()):
    print(f"    Cluster {c} : {n} data")

# ── 4. FUZZY C-MEANS ──────────────────────────────────────
print("\n" + "=" * 60)
print("  [3] FUZZY C-MEANS")
print("=" * 60)

X_T = X.T
cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
    X_T, c=K, m=2.0, error=0.005, maxiter=1000, seed=42
)
label_fcm = np.argmax(u, axis=0)
sil_fcm   = silhouette_score(X, label_fcm)
df["cluster_fuzzy"] = label_fcm

print(f"  Silhouette Score : {sil_fcm:.4f}")
print(f"  Distribusi cluster:")
for c, n in sorted(pd.Series(label_fcm).value_counts().items()):
    print(f"    Cluster {c} : {n} data")

# ── 5. PERBANDINGAN SILHOUETTE SCORE ──────────────────────
print("\n" + "=" * 60)
print("  PERBANDINGAN SILHOUETTE SCORE")
print("=" * 60)

hasil   = {"K-Means": sil_km, "K-Medoids": sil_kmd, "Fuzzy C-Means": sil_fcm}
terbaik = max(hasil, key=hasil.get)

for nama, skor in hasil.items():
    ket = " <- terbaik" if nama == terbaik else ""
    print(f"  {nama:<18} : {skor:.4f}{ket}")

print(f"\n  Interpretasi Silhouette Score:")
print(f"    0.71 - 1.00  : Struktur cluster sangat kuat")
print(f"    0.51 - 0.70  : Struktur cluster cukup kuat")
print(f"    0.26 - 0.50  : Struktur cluster lemah")
print(f"    < 0.25       : Tidak ada struktur yang jelas")

# ── 6. SIMPAN CSV DULU SEBELUM TAMPILKAN GRAFIK ───────────
df.to_csv("results_clustered.csv", index=False)
print(f"\n  Dataset tersimpan : results_clustered.csv")
print(f"  Kolom baru        : cluster_kmeans, cluster_kmedoid, cluster_fuzzy")

# ── 7. VISUALISASI ────────────────────────────────────────
WARNA = ["#2563EB","#16A34A","#DC2626","#D97706","#7C3AED"]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Hasil Clustering — K-Means vs K-Medoids vs Fuzzy C-Means",
             fontsize=14, fontweight="bold", y=1.02)

data_plot = [
    ("K-Means",        label_km,  sil_km,  kmeans.cluster_centers_),
    ("K-Medoids",       label_kmd, sil_kmd, centers_kmd),
    ("Fuzzy C-Means", label_fcm, sil_fcm, cntr),
]

for ax, (judul, labels, skor, centers) in zip(axes, data_plot):
    for i in range(K):
        mask = labels == i
        ax.scatter(X[mask, 0], X[mask, 1],
                   c=WARNA[i], label=f"Cluster {i}", alpha=0.6, s=30)
    ax.scatter(centers[:, 0], centers[:, 1],
               c="black", marker="X", s=150, zorder=5, label="Centroid")
    ax.set_title(f"{judul}\nSilhouette: {skor:.4f}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Rating (norm)", fontsize=10)
    ax.set_ylabel("Reviews (norm)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("clustering_hasil.png", dpi=150, bbox_inches="tight")
print(f"  Grafik tersimpan  : clustering_hasil.png")
plt.show()