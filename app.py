"""
=============================================================
  APLIKASI STREAMLIT - KLASTERISASI WISATA JAKARTA
  Algoritma : K-Means (5 Klaster)
  Jalankan  : streamlit run app.py
=============================================================
"""

import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
import plotly.express as px
from sklearn.metrics import silhouette_score

# ── SVG ICONS ─────────────────────────────────────────────
def svg_icon(name, size=20):
    icons = {
        "map":      f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"></path><circle cx="12" cy="10" r="3"></circle></svg>',
        "search":   f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"></circle><path d="m21 21-4.35-4.35"></path></svg>',
        "chart":    f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="20" x2="18" y2="10"></line><line x1="12" y1="20" x2="12" y2="4"></line><line x1="6" y1="20" x2="6" y2="14"></line></svg>',
        "table":    f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><line x1="3" y1="9" x2="21" y2="9"></line><line x1="3" y1="15" x2="21" y2="15"></line><line x1="9" y1="3" x2="9" y2="21"></line><line x1="15" y1="3" x2="15" y2="21"></line></svg>',
        "download": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg>',
    }
    return icons.get(name, "")

# ── KONFIGURASI HALAMAN ───────────────────────────────────
st.set_page_config(
    page_title="Wisata Jakarta - K-Means Clustering",
    page_icon="🗺️",
    layout="wide",
)

# ── KONSTANTA KLASTER ─────────────────────────────────────
# Berdasarkan analisis karakteristik tiap klaster:
# Klaster 2 → reviews & rating tertinggi  → Wisata Sangat Populer
# Klaster 1 → rating tinggi, reviews sedang → Wisata Populer
# Klaster 0 → jumlah terbanyak, rating sedang → Wisata Ramai
# Klaster 3 → rating & reviews rendah-sedang → Wisata Standar
# Klaster 4 → rating & reviews terendah → Wisata Kurang Dikenal
LABEL = {
    2: "Rating Bagus & Wisata Populer",
    1: "Rating Tinggi & Wisata Potensial",
    0: "Rating Standar & Wisata Ramai",
    3: "Rating dibawah Standar & Wisata Ramai",
    4: "Rating dibawah Standar & Wisata Kurang Ramai",
}

WARNA = {
    2: "#DC2626",   # merah   – sangat populer
    1: "#2563EB",   # biru    – populer
    0: "#16A34A",   # hijau   – ramai
    3: "#D97706",   # kuning  – standar
    4: "#6B7280",   # abu     – kurang dikenal
}

URUTAN_KLASTER = [2, 1, 0, 3, 4]

# ── LOAD & MERGE DATA ─────────────────────────────────────
@st.cache_data
def load_data():
    # Load data asli (rating & reviews belum ternormalisasi)
    wisata    = pd.read_csv("results_wisata.csv")
    clustered = pd.read_csv("results_clustered.csv")

    # Gabungkan berdasarkan posisi baris (bukan merge on="name")
    # Catatan: results_wisata.csv dan results_clustered.csv memiliki jumlah
    # baris dan urutan yang identik (sama-sama diturunkan dari
    # results_normalisasi.csv), sehingga penggabungan cukup dilakukan
    # berdasarkan index. Menggunakan merge(on="name") menyebabkan duplikasi
    # baris (521 -> 523) karena ada nama tempat yang sama
    # ("Museum Sejarah Jakarta") namun merepresentasikan dua lokasi/data
    # berbeda, sehingga merge menghasilkan cartesian product (2x2 baris).
    df = wisata.copy()
    df["cluster_kmeans"]  = clustered["cluster_kmeans"].values
    df["cluster_kmedoid"] = clustered["cluster_kmedoid"].values
    df["cluster_fuzzy"]   = clustered["cluster_fuzzy"].values

    # Tambah kolom label klaster
    df["Klaster K-Means"] = df["cluster_kmeans"].map(LABEL)

    # Rename kolom untuk tampilan UI
    df = df.rename(columns={
        "name"     : "Nama Tempat",
        "rating"   : "Rating",
        "reviews"  : "Ulasan",
        "url"      : "URL",
        "category" : "Kategori Wisata",
        "location" : "Lokasi",
    })

    # Hitung silhouette score dari data ternormalisasi
    norm = clustered[["rating", "reviews"]].values
    labels = clustered["cluster_kmeans"].values
    try:
        sil = round(float(silhouette_score(norm, labels)), 4)
    except Exception:
        sil = 0.0

    return df, sil

df, sil_score = load_data()

# ── HEADER ────────────────────────────────────────────────
st.markdown(
    "<h1><span style='display:inline-block;margin-right:10px'>"
    + svg_icon("map", 32)
    + "</span>Klasterisasi Tempat Wisata Jakarta</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    f"**Algoritma:** K-Means (k=5) &nbsp;|&nbsp; "
    f"**Silhouette Score:** {sil_score} &nbsp;|&nbsp; "
    f"**Total Data:** {len(df):,} tempat wisata"
)
st.divider()

# ── SIDEBAR FILTER ────────────────────────────────────────
st.sidebar.markdown(
    "<h2><span style='display:inline-block;margin-right:10px'>"
    + svg_icon("search", 24)
    + "</span>Filter Data</h2>",
    unsafe_allow_html=True,
)

label_options  = ["Semua Klaster"]  + [LABEL[k] for k in URUTAN_KLASTER]
lokasi_options = ["Semua Lokasi"]   + sorted(df["Lokasi"].dropna().unique().tolist())
kat_options    = ["Semua Kategori"] + sorted(df["Kategori Wisata"].dropna().unique().tolist())

pilih_klaster  = st.sidebar.selectbox("Pilih Klaster",         label_options)
pilih_lokasi   = st.sidebar.selectbox("Pilih Lokasi",          lokasi_options)
pilih_kategori = st.sidebar.selectbox("Pilih Kategori Wisata", kat_options)
cari           = st.sidebar.text_input("Cari Nama Tempat")

# ── FILTER DATA ───────────────────────────────────────────
df_filter = df.copy()
if pilih_klaster  != "Semua Klaster":
    df_filter = df_filter[df_filter["Klaster K-Means"]  == pilih_klaster]
if pilih_lokasi   != "Semua Lokasi":
    df_filter = df_filter[df_filter["Lokasi"]           == pilih_lokasi]
if pilih_kategori != "Semua Kategori":
    df_filter = df_filter[df_filter["Kategori Wisata"]  == pilih_kategori]
if cari:
    df_filter = df_filter[
        df_filter["Nama Tempat"].str.contains(cari, case=False, na=False)
    ]

# ── METRIK ────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Tempat",       f"{len(df_filter):,}")
col2.metric("Rata-rata Rating",   f"{df_filter['Rating'].mean():.2f} ⭐" if len(df_filter) > 0 else "-")
col3.metric("Total Ulasan",       f"{int(df_filter['Ulasan'].sum()):,}" if len(df_filter) > 0 else "-")
col4.metric("Jumlah Klaster",     f"{df_filter['cluster_kmeans'].nunique()}")
st.divider()

# ── TAB ───────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🗺️ Peta Sebaran", "📊 Grafik", "📋 Tabel Data"])

# ── TAB 1: PETA ───────────────────────────────────────────
with tab1:
    st.subheader("Peta Sebaran Tempat Wisata Jakarta")
    st.caption(f"Menampilkan {len(df_filter)} tempat wisata — klik titik untuk detail")

    m = folium.Map(location=[-6.2, 106.83], zoom_start=11, tiles="CartoDB positron")

    for _, row in df_filter.iterrows():
        if pd.notna(row["latitude"]) and pd.notna(row["longitude"]):
            cluster_id = int(row["cluster_kmeans"])
            warna_hex  = WARNA.get(cluster_id, "#999999")
            gmaps_url  = row["URL"] if pd.notna(row.get("URL")) \
                         else f"https://www.google.com/maps?q={row['latitude']},{row['longitude']}"

            popup_html = f"""
            <div style='width:220px;font-family:sans-serif'>
                <b style='font-size:13px'>{row['Nama Tempat']}</b><br>
                <hr style='margin:4px 0'>
                <b>Klaster:</b> {row['Klaster K-Means']}<br>
                <b>Rating:</b> ⭐ {row['Rating']}<br>
                <b>Ulasan:</b> {int(row['Ulasan']):,}<br>
                <b>Lokasi:</b> {row['Lokasi']}<br>
                <b>Kategori:</b> {row['Kategori Wisata']}<br>
                <hr style='margin:6px 0'>
                <a href='{gmaps_url}' target='_blank'
                   style='display:block;text-align:center;background:#1A73E8;
                          color:white;padding:6px;border-radius:6px;
                          text-decoration:none;font-weight:bold;font-size:12px'>
                    Buka di Google Maps
                </a>
            </div>
            """
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=6,
                color=warna_hex,
                fill=True,
                fill_color=warna_hex,
                fill_opacity=0.8,
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=row["Nama Tempat"],
            ).add_to(m)

    st_folium(m, width=None, height=520)

    # Legenda klaster (dirender sebagai elemen Streamlit native,
    # karena injeksi HTML custom ke folium.Element tidak konsisten
    # dirender oleh st_folium)
    st.markdown("**Klaster**")
    legend_cols = st.columns(len(URUTAN_KLASTER))
    for col, k in zip(legend_cols, URUTAN_KLASTER):
        col.markdown(
            f"<span style='color:{WARNA[k]};font-size:18px'>&#9679;</span> {LABEL[k]}",
            unsafe_allow_html=True,
        )

# ── TAB 2: GRAFIK ─────────────────────────────────────────
with tab2:
    st.subheader("Distribusi Klaster K-Means")

    color_map   = {v: WARNA[k] for k, v in LABEL.items()}
    urutan_label = [LABEL[k] for k in URUTAN_KLASTER]

    dist = (
        df_filter.groupby("Klaster K-Means")
        .size()
        .reindex(urutan_label)
        .dropna()
        .reset_index(name="Jumlah")
    )
    avg = (
        df_filter.groupby("Klaster K-Means")
        .agg(Avg_Rating=("Rating", "mean"), Avg_Ulasan=("Ulasan", "mean"))
        .reindex(urutan_label)
        .dropna()
        .reset_index()
    )

    col_a, col_b = st.columns(2)
    with col_a:
        fig1 = px.bar(
            dist, x="Klaster K-Means", y="Jumlah",
            color="Klaster K-Means", color_discrete_map=color_map,
            title="Jumlah Tempat Wisata per Klaster",
            text="Jumlah",
            category_orders={"Klaster K-Means": urutan_label},
        )
        fig1.update_traces(textposition="outside")
        fig1.update_layout(showlegend=False, xaxis_tickangle=-20)
        st.plotly_chart(fig1, use_container_width=True)

    with col_b:
        fig2 = px.pie(
            dist, names="Klaster K-Means", values="Jumlah",
            color="Klaster K-Means", color_discrete_map=color_map,
            title="Proporsi Klaster",
        )
        fig2.update_traces(textposition="inside", textinfo="percent+label")
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        fig3 = px.bar(
            avg, x="Klaster K-Means", y="Avg_Rating",
            color="Klaster K-Means", color_discrete_map=color_map,
            title="Rata-rata Rating per Klaster",
            text=avg["Avg_Rating"].round(2),
            category_orders={"Klaster K-Means": urutan_label},
        )
        fig3.update_traces(textposition="outside")
        fig3.update_layout(showlegend=False, xaxis_tickangle=-20, yaxis_range=[0, 5.5])
        st.plotly_chart(fig3, use_container_width=True)

    with col_d:
        fig4 = px.bar(
            avg, x="Klaster K-Means", y="Avg_Ulasan",
            color="Klaster K-Means", color_discrete_map=color_map,
            title="Rata-rata Jumlah Ulasan per Klaster",
            text=avg["Avg_Ulasan"].round(0),
            category_orders={"Klaster K-Means": urutan_label},
        )
        fig4.update_traces(textposition="outside")
        fig4.update_layout(showlegend=False, xaxis_tickangle=-20)
        st.plotly_chart(fig4, use_container_width=True)

# ── TAB 3: TABEL ──────────────────────────────────────────
with tab3:
    st.subheader(f"Tabel Data Tempat Wisata ({len(df_filter)} tempat)")

    tabel_tampil = df_filter[[
        "Nama Tempat", "Rating", "Ulasan",
        "Kategori Wisata", "Lokasi", "Klaster K-Means", "URL",
    ]].reset_index(drop=True)
    tabel_tampil.index += 1

    st.dataframe(
        tabel_tampil,
        use_container_width=True,
        height=450,
        column_config={
            "Rating" : st.column_config.NumberColumn("Rating ⭐", format="%.1f"),
            "Ulasan" : st.column_config.NumberColumn("Ulasan", format="%d"),
            "URL"    : st.column_config.LinkColumn(
                "Google Maps",
                display_text="Buka Maps",
                help="Klik untuk membuka lokasi di Google Maps",
            ),
        },
    )

    csv = tabel_tampil.drop(columns=["URL"]).to_csv(index=True).encode("utf-8")
    st.download_button(
        label="⬇️ Download Data (CSV)",
        data=csv,
        file_name="wisata_jakarta_kmeans.csv",
        mime="text/csv",
    )