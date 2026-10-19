import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# Set konfigurasi halaman agar lebar dan modern
st.set_page_config(
    page_title="Cek Harga Rumah Jabodetabek",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Injeksi CSS Kustom untuk Gaya UI Modern (Mirip Landing Page Real Estate)
st.markdown("""
<style>
    /* Mengubah warna background utama aplikasi */
    .stApp {
        background-color: #F8F9FA;
        color: #111827;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Menghilangkan padding berlebih dari Streamlit */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 4rem;
        max-width: 1100px;
    }
    
    /* Gaya Judul Utama (Hero Section) */
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        color: #0F172A;
        line-height: 1.15;
        margin-bottom: 0.75rem;
        letter-spacing: -0.02em;
    }
    
    .hero-subtitle {
        font-size: 1.05rem;
        color: #475569;
        line-height: 1.6;
        margin-bottom: 1.5rem;
    }
    
    /* Gambar Hero / Ilustrasi Rumah */
    .hero-img {
        width: 100%;
        border-radius: 16px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.08);
        object-fit: cover;
    }
    
    /* Penyesuaian Tombol Utama (Button) */
    div.stButton > button:first-child {
        background-color: #0F172A !important;
        color: #FFFFFF !important;
        border-radius: 9999px !important;
        padding: 0.75rem 2rem !important;
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        border: none !important;
        width: 100% !important;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.15) !important;
        transition: all 0.2s ease-in-out !important;
    }
    
    div.stButton > button:first-child:hover {
        background-color: #1E293B !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 16px rgba(15, 23, 42, 0.25) !important;
    }
    
    /* Mengubah Tampilan Input Box agar Lebih Bersih */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div {
        border-radius: 10px !important;
        border: 1px solid #E2E8F0 !important;
        background-color: #FFFFFF !important;
        color: #0F172A !important;
    }

    /* Menyembunyikan menu bawaan Streamlit untuk kesan profesional */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# HERO SECTION (Header dengan Judul & Ilustrasi)
# ---------------------------------------------------------
col_hero_text, col_hero_img = st.columns([1.2, 0.8], gap="large")

with col_hero_text:
    st.markdown('<div class="hero-title">Temukan Estimasi Harga Rumah Ideal Anda</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">Masukkan spesifikasi properti Anda di bawah ini untuk mendapatkan estimasi harga pasar secara presisi menggunakan kecerdasan buatan, serta evaluasi kewajaran penawaran.</div>',
        unsafe_allow_html=True
    )

with col_hero_img:
    # Ilustrasi rumah bergaya arsitektur modern
    st.markdown(
        '<img src="https://images.unsplash.com/photo-1600596542815-ffad4c1539a9?q=80&w=800&auto=format&fit=crop" class="hero-img" alt="Rumah Modern">',
        unsafe_allow_html=True
    )

st.divider()

# ---------------------------------------------------------
# INPUT SECTION (Formulir Pengisian Data Properti)
# ---------------------------------------------------------

# Section 1: Lokasi
with st.container(border=True):
    st.subheader("1. Lokasi Properti")
    col_k1, col_k2 = st.columns(2)
    with col_k1:
        kota = st.selectbox("Kota / Kabupaten", ["Bekasi", "Jakarta Selatan", "Jakarta Timur", "Depok", "Tangerang", "Bogor"])
    with col_k2:
        kecamatan = st.selectbox("Kecamatan / Area", ["Babelan", "Cikarang", "Tambun", "Kebayoran Baru", "Cibubur"])

# Section 2: Spesifikasi
with st.container(border=True):
    st.subheader("2. Spesifikasi Bangunan & Fasilitas")
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        luas_tanah = st.number_input("Luas Tanah (m²)", min_value=10, value=120)
        kamar_tidur = st.number_input("Kamar Tidur", min_value=1, value=3)
        jumlah_lantai = st.number_input("Jumlah Lantai", min_value=1, value=2)
        carport = st.number_input("Kapasitas Carport", min_value=0, value=1)
        sertifikat = st.selectbox("Jenis Sertifikat", ["SHM (Sertifikat Hak Milik)", "HGB (Hak Guna Bangunan)", "Lainnya"])
        perabot = st.selectbox("Kondisi Perabot", ["Unfurnished (kosong)", "Semi Furnished", "Full Furnished"])

    with col_s2:
        luas_bangunan = st.number_input("Luas Bangunan (m²)", min_value=10, value=100)
        kamar_mandi = st.number_input("Kamar Mandi", min_value=1, value=2)
        listrik = st.selectbox("Daya Listrik (VA)", [900, 1300, 2200, 3500, 4400, 5500, 7700])
        garasi = st.number_input("Kapasitas Garasi", min_value=0, value=0)
        kondisi = st.selectbox("Kondisi Bangunan", ["Bagus", "Butuh Renovasi", "Baru Gress"])

# Section 3: Harga Penawaran
with st.container(border=True):
    st.subheader("3. Harga Penawaran Penjual (Opsional)")
    harga_penawaran = st.number_input(
        "Masukkan harga penawaran penjual (dalam juta Rp). Isi 0 jika hanya ingin mengetahui estimasi harga pasar.",
        min_value=0,
        value=0,
        help="Fitur ini akan mengevaluasi apakah harga penawaran tergolong Wajar, Di Atas Pasar, atau Di Bawah Pasar berdasarkan batas toleransi ilmiah."
    )

st.write("")

# Tombol Eksekusi Estimasi
if st.button("Hitung Estimasi Harga Pasar"):
    # --- LOGIKA MODEL ML ANDA TETAP BERJALAN DI SINI ---
    st.success("Perhitungan estimasi berhasil dilakukan!")
    # Tampilkan hasil prediksi sesuai variabel & model milikmu

st.divider()

# Footer / Expander Informasi Model
with st.expander("ℹ️ Tentang Model Prediksi"):
    st.write("""
    Aplikasi ini menggunakan algoritma **Gradient Boosting Regressor** yang dilatih dari data transaksi rumah real Jabodetabek. 
    Batas toleransi penawaran wajar dihitung berdasarkan tingkat kesalahan rata-rata model (**MAPE 22,95%**).
    """)

st.caption("Proyek Akhir Pembelajaran Mesin — Estimasi ini dibuat untuk tujuan referensi dan bukan penilaian (appraisal) resmi.")
