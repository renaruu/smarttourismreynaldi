"""
=============================================================
  ELBOW METHOD — MENENTUKAN NILAI K OPTIMAL
  File sumber : results_normalisasi.csv
  Fitur       : rating, reviews (sudah ternormalisasi)
  Tujuan      : menentukan k untuk KMeans, KMedoid, Fuzzy C-Means
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# ── 1. LOAD DATA ──────────────────────────────────────────
FILE_IN = "results_normalisasi.csv"

df = pd.read_csv(FILE_IN)
X  = df[["rating", "reviews"]].values

print("=" * 55)
print("  ELBOW METHOD")
print("=" * 55)
print(f"  Jumlah data : {len(X):,}")
print(f"  Fitur       : rating, reviews")
print(f"  Range k     : 2 – 10")

# ── 2. HITUNG WCSS TIAP NILAI K ───────────────────────────
wcss = []
K_RANGE = range(2, 11)

print("\n  Menghitung WCSS per nilai k...")
print("-" * 35)

for k in K_RANGE:
    kmeans = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    print(f"  k = {k}  ->  WCSS = {kmeans.inertia_:.6f}")

# ── 3. DETEKSI SIKU OTOMATIS (KNEEDLE) ────────────────────
delta1   = np.diff(wcss)
delta2   = np.diff(delta1)
k_optimal = int(K_RANGE[np.argmax(delta2) + 2])

print("\n" + "=" * 55)
print(f"  K OPTIMAL (titik siku)  : {k_optimal}")
print("=" * 55)

# ── 4. PLOT GRAFIK ELBOW ──────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(list(K_RANGE), wcss,
        marker="o", color="#2563EB", linewidth=2.5,
        markersize=8, markerfacecolor="white", markeredgewidth=2,
        label="WCSS")

ax.axvline(x=k_optimal, color="#DC2626", linestyle="--",
           linewidth=1.8, label=f"k optimal = {k_optimal}")
ax.scatter([k_optimal], [wcss[k_optimal - 2]],
           color="#DC2626", s=120, zorder=5)

ax.annotate(f"  k = {k_optimal}\n  WCSS = {wcss[k_optimal - 2]:.4f}",
            xy=(k_optimal, wcss[k_optimal - 2]),
            xytext=(k_optimal + 0.4, wcss[k_optimal - 2] + 0.002),
            fontsize=10, color="#DC2626")

ax.set_title("Elbow Method — Penentuan Nilai K Optimal",
             fontsize=14, fontweight="bold", pad=15)
ax.set_xlabel("Jumlah Cluster (k)", fontsize=12)
ax.set_ylabel("WCSS (Within-Cluster Sum of Squares)", fontsize=12)
ax.set_xticks(list(K_RANGE))
ax.legend(fontsize=11)
ax.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("elbow_plot.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"\n  Grafik tersimpan : elbow_plot.png")
print(f"  Gunakan k = {k_optimal} untuk KMeans, KMedoid, dan Fuzzy C-Means")