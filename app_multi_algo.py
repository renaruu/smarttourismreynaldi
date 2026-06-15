"""
=============================================================
  APLIKASI STREAMLIT - KLASTERISASI WISATA JAKARTA
  Algoritma : K-Means, K-Medoids, Fuzzy C-Means, HDBSCAN, OPTICS
  Status    : CADANGAN - Multi Algoritma
  Jalankan  : streamlit run app_multi_algo.py
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
    """Return SVG icon by name"""
    icons = {
        "map": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"></path><circle cx="12" cy="10" r="3"></circle></svg>',
        "settings": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"></circle><path d="M12 1v6m0 6v6M4.22 4.22l4.24 4.24m5.08 5.08l4.24 4.24M1 12h6m6 0h6M4.22 19.78l4.24-4.24m5.08-5.08l4.24-4.24"></path></svg>',
        "search": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"></circle><path d="m21 21-4.35-4.35"></path></svg>',
        "info": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><path d="M12 16v-4M12 8h.01"></path></svg>',
        "target": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="1"></circle><circle cx="12" cy="12" r="5"></circle><circle cx="12" cy="12" r="9"></circle></svg>',
        "chart": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="12" y1="2" x2="12" y2="22"></line><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"></path></svg>',
        "table": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><line x1="3" y1="9" x2="21" y2="9"></line><line x1="3" y1="15" x2="21" y2="15"></line><line x1="9" y1="3" x2="9" y2="21"></line><line x1="15" y1="3" x2="15" y2="21"></line></svg>',
        "trending": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"></polyline><polyline points="17 6 23 6 23 12"></polyline></svg>',
        "download": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg>',
        "location": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"></path><circle cx="12" cy="10" r="3"></circle></svg>',
        "star": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="currentColor"><polygon points="12 2 15.09 10.26 23.77 10.5 17.39 16.99 19.68 25.58 12 20.16 4.32 25.58 6.61 16.99 0.23 10.5 8.91 10.26 12 2"/></svg>',
        "comment": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path></svg>',
        "tag": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20.59 13.41l-7.17 7.17a2 2 0 0 1-2.83 0L2 12V2h10l8.59 8.59a2 2 0 0 1 0 2.82z"></path><line x1="7" y1="7" x2="7.01" y2="7"></line></svg>',
        "category": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="7" height="7"></rect><rect x="14" y="3" width="7" height="7"></rect><rect x="14" y="14" width="7" height="7"></rect><rect x="3" y="14" width="7" height="7"></rect></svg>',
        "sparkle": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="currentColor"><polygon points="12 2 15.09 10.26 23.77 10.5 17.39 16.99 19.68 25.58 12 20.16 4.32 25.58 6.61 16.99 0.23 10.5 8.91 10.26 12 2"/></svg>',
    }
    return icons.get(name, "")

# ── KONFIGURASI HALAMAN ───────────────────────────────────
st.set_page_config(
    page_title="Wisata Jakarta - Multi Algoritma Clustering",
    page_icon="�",
    layout="wide",
)

# ── LOAD DATA ─────────────────────────────────────────────
@st.cache_data
def load_data():
    """Load semua data dan hitung Silhouette Score untuk setiap algoritma"""
    df_wisata = pd.read_csv("results_wisata.csv")
    df_km     = pd.read_csv("results_clustered.csv")
    df_hdb    = pd.read_csv("results_hdbscan_optics.csv")
    df_norm   = pd.read_csv("results_normalisasi.csv")
    
    # Mapping kolom dari hasil clustering
    df_wisata["cluster_kmeans"]  = df_km["cluster_kmeans"].values
    df_wisata["cluster_kmedoid"] = df_km["cluster_kmedoid"].values
    df_wisata["cluster_fuzzy"]   = df_km["cluster_fuzzy"].values
    df_wisata["cluster_hdbscan"] = df_hdb["cluster_hdbscan"].values
    df_wisata["cluster_optics"]  = df_hdb["cluster_optics"].values
    
    # Hitung Silhouette Score
    X = df_norm[["rating", "reviews"]].values
    
    def hitung_sil(labels):
        mask = labels != -1
        if len(set(labels[mask])) > 1:
            return round(silhouette_score(X[mask], labels[mask]), 4)
        return 0.0
    
    sil_scores = {
        "K-Means": hitung_sil(df_wisata["cluster_kmeans"].values),
        "K-Medoids": hitung_sil(df_wisata["cluster_kmedoid"].values),
        "Fuzzy C-Means": hitung_sil(df_wisata["cluster_fuzzy"].values),
        "HDBSCAN": hitung_sil(df_wisata["cluster_hdbscan"].values),
        "OPTICS": hitung_sil(df_wisata["cluster_optics"].values),
    }
    
    return df_wisata, sil_scores

df, sil_scores = load_data()

# ── MAPPING ALGORITMA ─────────────────────────────────────
ALGO_CONFIG = {
    "K-Means": {
        "kolom": "cluster_kmeans",
        "icon": "🎯",
        "deskripsi": "Partisi-based clustering dengan K centroid"
    },
    "K-Medoids": {
        "kolom": "cluster_kmedoid",
        "icon": "💠",
        "deskripsi": "Robust variant dari K-Means menggunakan medoid"
    },
    "Fuzzy C-Means": {
        "kolom": "cluster_fuzzy",
        "icon": "🌊",
        "deskripsi": "Soft clustering dengan membership probability"
    },
    "HDBSCAN": {
        "kolom": "cluster_hdbscan",
        "icon": "🌳",
        "deskripsi": "Density-based clustering dengan hierarchical approach"
    },
    "OPTICS": {
        "kolom": "cluster_optics",
        "icon": "🔍",
        "deskripsi": "Ordering Points To Identify Clustering Structure"
    }
}

# ── WARNA KLASTER ─────────────────────────────────────────
WARNA = {
    0: "#16A34A",
    1: "#2563EB", 
    2: "#7C3AED",
    3: "#0891B2",
    4: "#D97706",
    5: "#BE185D",
    6: "#6B7280",
    7: "#DC2626",
    -1: "#999999",  # Noise/Outlier
}

# ── HEADER ────────────────────────────────────────────────
st.markdown("<h1><span style='display:inline-block;margin-right:10px'>" + svg_icon('map', 32) + "</span>Klasterisasi Tempat Wisata Jakarta - Multi Algoritma</h1>", unsafe_allow_html=True)
st.markdown("**Perbandingan 5 Algoritma Clustering untuk Segmentasi Tempat Wisata**")

# ── SIDEBAR - SELECTOR ALGORITMA ──────────────────────────
st.sidebar.markdown("<h2><span style='display:inline-block;margin-right:10px'>" + svg_icon('settings', 24) + "</span>Konfigurasi</h2>", unsafe_allow_html=True)

pilih_algo = st.sidebar.selectbox(
    "Pilih Algoritma Clustering",
    options=list(ALGO_CONFIG.keys())
)

algo_info = ALGO_CONFIG[pilih_algo]
kolom_klaster = algo_info["kolom"]
sil_score = sil_scores[pilih_algo]

# Tampilkan info algoritma
with st.sidebar.expander("Info Algoritma"):
    st.write(f"**{pilih_algo}**")
    st.caption(algo_info["deskripsi"])
    st.metric("Silhouette Score", f"{sil_score:.4f}")

# ── SIDEBAR FILTER ────────────────────────────────────────
st.sidebar.markdown("<h2><span style='display:inline-block;margin-right:10px'>" + svg_icon('search', 24) + "</span>Filter Data</h2>", unsafe_allow_html=True)

# Buat dataframe dengan kolom klaster yang dipilih
df_work = df.copy()
df_work["No Klaster"] = df_work[kolom_klaster]

# Cek apakah ada noise untuk algoritma ini
has_noise = -1 in df_work["No Klaster"].values

# Build cluster list dengan opsi Noise/Outlier jika ada
clusters = sorted([c for c in df_work["No Klaster"].unique() if c != -1])
semua_klaster  = ["Semua Klaster"] + clusters
if has_noise:
    semua_klaster.append("Noise/Outlier")

semua_lokasi   = ["Semua Lokasi"]   + sorted(df_work["location"].dropna().unique().tolist())
semua_kategori = ["Semua Kategori"] + sorted(df_work["category"].dropna().unique().tolist())

pilih_klaster  = st.sidebar.selectbox("Pilih Klaster", semua_klaster)
pilih_lokasi   = st.sidebar.selectbox("Pilih Lokasi", semua_lokasi)
pilih_kategori = st.sidebar.selectbox("Pilih Kategori Wisata", semua_kategori)
cari           = st.sidebar.text_input("Cari Nama Tempat")

# ── FILTER DATA ───────────────────────────────────────────
df_filter = df_work.copy()
if pilih_klaster != "Semua Klaster":
    if pilih_klaster == "Noise/Outlier":
        df_filter = df_filter[df_filter["No Klaster"] == -1]
    else:
        df_filter = df_filter[df_filter["No Klaster"] == int(pilih_klaster)]
if pilih_lokasi != "Semua Lokasi":
    df_filter = df_filter[df_filter["location"] == pilih_lokasi]
if pilih_kategori != "Semua Kategori":
    df_filter = df_filter[df_filter["category"] == pilih_kategori]
if cari:
    df_filter = df_filter[df_filter["name"].str.contains(cari, case=False, na=False)]

# Pisahkan noise dan non-noise untuk analisis
df_noise = df_work[df_work["No Klaster"] == -1].copy()

# ── HEADER DENGAN METRIK ──────────────────────────────────
col_header = st.columns([2, 1])
with col_header[0]:
    st.markdown(f"### <span style='display:inline-block;margin-right:10px'>{svg_icon('target', 28)}</span>{pilih_algo}", unsafe_allow_html=True)
with col_header[1]:
    st.metric("Silhouette", f"{sil_score:.4f}")

st.divider()

# ── METRIK ────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Tempat", f"{len(df_filter):,}")
col2.metric("Rata-rata Rating", f"{df_filter['rating'].mean():.2f}" if len(df_filter) > 0 else "-")
col3.metric("Total Ulasan", f"{int(df_filter['reviews'].sum()):,}" if len(df_filter) > 0 else "-")
col4.metric("Jumlah Klaster", f"{df_filter['No Klaster'].nunique()}")
col5.metric("Noise/Outlier", f"{len(df_noise):,}" if has_noise else "Tidak ada")

st.divider()

# ── TAB ───────────────────────────────────────────────────
if has_noise:
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Peta Sebaran", "Grafik", "Tabel Data", "Perbandingan", "Analisis Noise"])
else:
    tab1, tab2, tab3, tab4 = st.tabs(["Peta Sebaran", "Grafik", "Tabel Data", "Perbandingan"])

# ── TAB 1: PETA ───────────────────────────────────────────
with tab1:
    st.markdown(f"### <span style='display:inline-block;margin-right:10px'>{svg_icon('map', 28)}</span>Peta Sebaran Tempat Wisata - {pilih_algo}", unsafe_allow_html=True)
    st.caption(f"Menampilkan {len(df_filter)} tempat wisata — klik titik untuk detail")

    m = folium.Map(location=[-6.2, 106.83], zoom_start=11, tiles="CartoDB positron")

    for _, row in df_filter.iterrows():
        if pd.notna(row["latitude"]) and pd.notna(row["longitude"]):
            warna_hex = WARNA.get(int(row["No Klaster"]), "#999999")
            gmaps_url = row["url"] if pd.notna(row.get("url", None)) else f"https://www.google.com/maps?q={row['latitude']},{row['longitude']}"
            
            popup_html = f"""
            <div style='width:220px;font-family:sans-serif'>
                <b style='font-size:13px'>{row['name']}</b><br>
                <hr style='margin:4px 0'>
                <b>Klaster:</b> {int(row['No Klaster'])}<br>
                <b>Rating:</b> {row['rating']:.2f}<br>
                <b>Ulasan:</b> {int(row['reviews']):,}<br>
                <b>Lokasi:</b> {row['location']}<br>
                <b>Kategori:</b> {row['category']}<br>
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
                tooltip=row["name"],
            ).add_to(m)

    st_folium(m, width=None, height=520)

    # Legenda klaster (dirender sebagai elemen Streamlit native,
    # karena injeksi HTML custom ke folium.Element tidak konsisten
    # dirender oleh st_folium)
    st.markdown(f"**Klaster ({pilih_algo})**")
    klaster_legend = sorted([c for c in df_work["No Klaster"].unique() if c != -1])
    if -1 in df_work["No Klaster"].unique():
        klaster_legend.append(-1)

    legend_cols = st.columns(len(klaster_legend))
    for col, k in zip(legend_cols, klaster_legend):
        label = "Noise/Outlier" if k == -1 else f"Klaster {int(k)}"
        col.markdown(
            f"<span style='color:{WARNA[int(k)]};font-size:18px'>&#9679;</span> {label}",
            unsafe_allow_html=True,
        )

# ── TAB 2: GRAFIK ─────────────────────────────────────────
with tab2:
    st.markdown(f"### <span style='display:inline-block;margin-right:10px'>{svg_icon('chart', 28)}</span>Analisis Grafik - {pilih_algo}", unsafe_allow_html=True)

    # Buat color map dinamis
    klaster_unik = sorted([c for c in df_filter["No Klaster"].unique()])
    color_map = {k: WARNA.get(int(k), "#999999") for k in klaster_unik}

    dist = df_filter.groupby("No Klaster").size().reset_index(name="Jumlah")
    avg  = df_filter.groupby("No Klaster").agg(
        Avg_Rating=("rating", "mean"),
        Avg_Ulasan=("reviews", "mean"),
    ).reset_index()

    col_a, col_b = st.columns(2)

    with col_a:
        fig1 = px.bar(
            dist, x="No Klaster", y="Jumlah",
            color="No Klaster", color_discrete_map=color_map,
            title="Jumlah Tempat Wisata Berdasarkan Klaster",
            text="Jumlah"
        )
        fig1.update_traces(textposition="outside")
        fig1.update_layout(showlegend=False, xaxis_tickangle=-30)
        st.plotly_chart(fig1, use_container_width=True)

    with col_b:
        fig2 = px.pie(
            dist, names="No Klaster", values="Jumlah",
            color="No Klaster", color_discrete_map=color_map,
            title="Proporsi Klaster"
        )
        fig2.update_traces(textposition="inside", textinfo="percent+label")
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    col_c, col_d = st.columns(2)

    with col_c:
        fig3 = px.bar(
            avg, x="No Klaster", y="Avg_Rating",
            color="No Klaster", color_discrete_map=color_map,
            title="Rata-rata Rating Berdasarkan Klaster",
            text=avg["Avg_Rating"].round(2)
        )
        fig3.update_traces(textposition="outside")
        fig3.update_layout(showlegend=False, xaxis_tickangle=-30, yaxis_range=[3.8, 5.2])
        st.plotly_chart(fig3, use_container_width=True)

    with col_d:
        fig4 = px.bar(
            avg, x="No Klaster", y="Avg_Ulasan",
            color="No Klaster", color_discrete_map=color_map,
            title="Rata-rata Ulasan Berdasarkan Klaster",
            text=avg["Avg_Ulasan"].round(0)
        )
        fig4.update_traces(textposition="outside")
        fig4.update_layout(showlegend=False, xaxis_tickangle=-30)
        st.plotly_chart(fig4, use_container_width=True)

# ── TAB 3: TABEL ──────────────────────────────────────────
with tab3:
    st.markdown(f"### <span style='display:inline-block;margin-right:10px'>{svg_icon('table', 28)}</span>Tabel Data Tempat Wisata ({len(df_filter)} tempat)", unsafe_allow_html=True)

    tabel_tampil = df_filter[[
        "name", "rating", "reviews",
        "category", "location", "No Klaster", "url"
    ]].copy()
    tabel_tampil.columns = ["Nama Tempat", "Rating", "Ulasan", "Kategori", "Lokasi", "Klaster", "URL"]
    tabel_tampil = tabel_tampil.reset_index(drop=True)
    tabel_tampil.index += 1

    st.dataframe(
        tabel_tampil,
        use_container_width=True,
        height=450,
        column_config={
            "Rating" : st.column_config.NumberColumn("Rating", format="%.2f"),
            "Ulasan" : st.column_config.NumberColumn("Ulasan", format="%d"),
            "Klaster": st.column_config.NumberColumn("Klaster", format="%d"),
            "URL"    : st.column_config.LinkColumn(
                "Google Maps",
                display_text="Buka Maps",
            ),
        }
    )

    # Download CSV
    csv = tabel_tampil.drop(columns=["URL"]).to_csv(index=True).encode("utf-8")
    st.download_button(
        label="Download Data (CSV)",
        data=csv,
        file_name=f"wisata_jakarta_{pilih_algo.lower().replace(' ', '_')}.csv",
        mime="text/csv"
    )

# ── TAB 4: PERBANDINGAN ───────────────────────────────────
with tab4:
    st.markdown(f"### <span style='display:inline-block;margin-right:10px'>{svg_icon('trending', 28)}</span>Perbandingan Silhouette Score - Semua Algoritma", unsafe_allow_html=True)
    
    # Tabel perbandingan
    comp_data = {
        "Algoritma": list(sil_scores.keys()),
        "Silhouette Score": list(sil_scores.values())
    }
    df_comp = pd.DataFrame(comp_data)
    df_comp = df_comp.sort_values("Silhouette Score", ascending=False)
    
    col_comp1, col_comp2 = st.columns(2)
    
    with col_comp1:
        st.dataframe(df_comp, use_container_width=True, hide_index=True)
    
    with col_comp2:
        fig_comp = px.bar(
            df_comp, 
            x="Algoritma", 
            y="Silhouette Score",
            color="Silhouette Score",
            color_continuous_scale="RdYlGn",
            text="Silhouette Score",
            title="Perbandingan Silhouette Score"
        )
        fig_comp.update_traces(texttemplate="%.4f", textposition="outside")
        fig_comp.update_layout(showlegend=False, xaxis_tickangle=-30)
        st.plotly_chart(fig_comp, use_container_width=True)
    

# ── TAB 5: ANALISIS NOISE (Jika ada noise) ────────────────
if has_noise:
    with tab5:
        st.markdown(f"### <span style='display:inline-block;margin-right:10px'>{svg_icon('search', 28)}</span>Analisis Noise/Outlier - {pilih_algo}", unsafe_allow_html=True)
        
        if len(df_noise) > 0:
            st.info(f"Ditemukan **{len(df_noise)} tempat wisata** yang teridentifikasi sebagai noise/outlier oleh algoritma {pilih_algo}.")
            
            # Statistik Noise
            st.subheader("Statistik Noise")
            col_noise1, col_noise2, col_noise3 = st.columns(3)
            
            col_noise1.metric("Total Noise", f"{len(df_noise)}")
            col_noise2.metric("Rata-rata Rating Noise", f"{df_noise['rating'].mean():.2f}")
            col_noise3.metric("Rata-rata Ulasan Noise", f"{int(df_noise['reviews'].mean()):,}")
            
            st.divider()
            
            # Tabel Noise
            st.subheader("Daftar Tempat Wisata Noise/Outlier")
            
            tabel_noise = df_noise[[
                "name", "rating", "reviews",
                "category", "location", "url"
            ]].copy()
            tabel_noise.columns = ["Nama Tempat", "Rating", "Ulasan", "Kategori", "Lokasi", "URL"]
            tabel_noise = tabel_noise.reset_index(drop=True)
            tabel_noise.index += 1
            
            st.dataframe(
                tabel_noise,
                use_container_width=True,
                height=450,
                column_config={
                    "Rating" : st.column_config.NumberColumn("Rating", format="%.2f"),
                    "Ulasan" : st.column_config.NumberColumn("Ulasan", format="%d"),
                    "URL"    : st.column_config.LinkColumn(
                        "Google Maps",
                        display_text="Buka Maps",
                    ),
                }
            )
            
            # Distribusi Noise berdasarkan Lokasi dan Kategori
            st.subheader("Distribusi Noise")
            col_dist1, col_dist2 = st.columns(2)
            
            with col_dist1:
                noise_by_location = df_noise['location'].value_counts().reset_index()
                noise_by_location.columns = ['Lokasi', 'Jumlah']
                fig_location = px.bar(
                    noise_by_location,
                    x='Lokasi',
                    y='Jumlah',
                    title='Noise Berdasarkan Lokasi',
                    text='Jumlah',
                    color='Jumlah',
                    color_continuous_scale='Reds'
                )
                fig_location.update_traces(textposition="outside")
                fig_location.update_layout(showlegend=False, xaxis_tickangle=-45)
                st.plotly_chart(fig_location, use_container_width=True)
            
            with col_dist2:
                noise_by_category = df_noise['category'].value_counts().reset_index()
                noise_by_category.columns = ['Kategori', 'Jumlah']
                fig_category = px.pie(
                    noise_by_category,
                    names='Kategori',
                    values='Jumlah',
                    title='Noise Berdasarkan Kategori'
                )
                fig_category.update_traces(textposition="inside", textinfo="percent+label")
                st.plotly_chart(fig_category, use_container_width=True)
            
            # Download Noise Data
            st.divider()
            csv_noise = tabel_noise.drop(columns=["URL"]).to_csv(index=True).encode("utf-8")
            st.download_button(
                label="Download Data Noise (CSV)",
                data=csv_noise,
                file_name=f"noise_{pilih_algo.lower().replace(' ', '_')}.csv",
                mime="text/csv"
            )
        else:
            st.success("Tidak ada noise/outlier yang ditemukan! Semua tempat wisata terklaster dengan baik.")

# ── FOOTER ────────────────────────────────────────────────
st.divider()
st.caption(f"Aplikasi Streamlit - Klasterisasi Wisata Jakarta | Algoritma: {pilih_algo} | Silhouette: {sil_score}")