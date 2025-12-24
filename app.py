import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

# =============================
# KONFIGURASI HALAMAN
# =============================
st.set_page_config(
    page_title="TA-12 Time Series",
    layout="wide"
)

st.title("📊 TA-12 – Eksplorasi Data Time Series Iklim")
st.markdown("Aplikasi eksplorasi data time series sesuai Modul Praktikum 12.")

# =============================
# LOAD DATA
# =============================
uploaded_file = st.file_uploader(
    "Upload Dataset Excel (.xlsx)",
    type=["xlsx"]
)

if uploaded_file is not None:

    # Load dataset
    df = pd.read_excel(uploaded_file)

    # Konversi tanggal
    df['Tanggal'] = pd.to_datetime(df['Tanggal'])
    df = df.set_index('Tanggal').sort_index()

    st.subheader("📄 Preview Data")
    st.dataframe(df.head())

    # =============================
    # CLEANING DATA
    # =============================
    numeric_cols = df.select_dtypes(include='number').columns
    df[numeric_cols] = df[numeric_cols].interpolate(method='time')
    df[numeric_cols] = df[numeric_cols].fillna(method='ffill').fillna(method='bfill')

    # =============================
    # PILIH VARIABEL
    # =============================
    st.sidebar.header("Pengaturan")
    variable = st.sidebar.selectbox(
        "Pilih Variabel Numerik",
        numeric_cols
    )

    # =============================
    # VISUALISASI TIME SERIES
    # =============================
    st.subheader("📈 Time Series Harian")
    fig1, ax1 = plt.subplots()
    ax1.plot(df[variable])
    ax1.set_xlabel("Tanggal")
    ax1.set_ylabel(variable)
    st.pyplot(fig1)

    # =============================
    # RESAMPLING
    # =============================
    st.subheader("🔁 Resampling Data")
    monthly = df[variable].resample('M').mean()
    yearly = df[variable].resample('Y').mean()

    fig2, ax2 = plt.subplots()
    ax2.plot(df[variable], alpha=0.4, label="Harian")
    ax2.plot(monthly, label="Bulanan")
    ax2.plot(yearly, label="Tahunan")
    ax2.legend()
    st.pyplot(fig2)

    # =============================
    # DEKOMPOSISI
    # =============================
    st.subheader("🧩 Dekomposisi Time Series")
    decomposition = seasonal_decompose(
        df[variable],
        model='additive',
        period=365
    )
    fig3 = decomposition.plot()
    st.pyplot(fig3)

    # =============================
    # UJI STASIONERITAS
    # =============================
    st.subheader("🧪 Uji Stasioneritas (ADF Test)")
    adf_result = adfuller(df[variable].dropna())

    st.write(f"ADF Statistic : {adf_result[0]:.4f}")
    st.write(f"p-value       : {adf_result[1]:.4f}")

    if adf_result[1] < 0.05:
        st.success("Data sudah stasioner")
    else:
        st.warning("Data belum stasioner")

else:
    st.info("Silakan upload dataset Excel untuk memulai analisis.")
