import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Call st.set_page_config as the very first Streamlit command
st.set_page_config(
    page_title="Regresi Linier",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None
)

# Define the sidebar menu
with st.sidebar:
    selected = option_menu("Menu Utama", ["Dashboard", "Visualisasi Data", "Perhitungan"], icons=['house', 'pie-chart'], menu_icon="cast", default_index=0)

# Define a function for the "Dashboard" page
def dashboard_page():
    st.markdown("---")
    st.title("Estimasi Harga Mobil Bekas Menggunakan Algoritma Regresi Linier")
    st.write("Estimasi harga mobil bekas menggunakan algoritma regresi linier adalah salah satu tugas umum dalam analisis data dan pemodelan prediktif. Regresi linier adalah metode statistik yang digunakan untuk mengukur hubungan antara satu atau lebih variabel independen (fitur) dengan variabel dependen (harga mobil bekas). ")
    st.markdown("---")

# Define a function for the "Visualisasi Data" page
def data_visualization_page():
    st.title('')
    
    # Judul Utama
    st.title("Analisis Berkas CSV")
    
    # Unggah berkas
    uploaded_file = st.file_uploader("Unggah berkas CSV", type=["csv"])
    
    data = None  # Inisialisasi data sebagai None
    
    if uploaded_file is not None:
        st.write("Berkas berhasil diunggah.")
        
        # Baca berkas CSV ke dalam DataFrame
        df = pd.read_csv(uploaded_file)
        
        # Tampilkan informasi dasar tentang data
        st.subheader("Ikhtisar Data")
        st.write("Jumlah baris:", df.shape[0])
        st.write("Jumlah kolom:", df.shape[1])
        
        # Tampilkan beberapa baris pertama data
        st.subheader("Pratinjau Data")
        st.dataframe(df.head())
        
        # Analisis data dan visualisasi dapat ditambahkan di sini
    
        # Contoh: Gambar diagram batang dari kolom tertentu
        st.subheader("Diagram Batang")
        st.write(" grafik batang yang menampilkan sebaran data dalam kolom yang dipilih.")
        selected_column = st.selectbox("Pilih kolom untuk diagram batang", df.columns, key="bar_chart_selectbox")
        st.bar_chart(df[selected_column])

        # Contoh: Tampilkan statistik ringkas
        st.subheader("Statistik Ringkas")
        st.write(" ringkasan statistik tentang data numerik dalam dataset. Statistik ini mencakup informasi seperti jumlah data, rata-rata, deviasi standar, nilai minimum, dan nilai maksimum.")
        st.write(df.describe(), key="summary_stats")
        
        if st.checkbox("Visualisasi data"):
            data = df  # Setel variabel data ke df yang baru dibaca    
    if data is not None:
        # Tampilkan data penduduk dalam bentuk tabel
        st.subheader('Estimasi Harga Mobil Bekas Menggunakan Algoritma Regresi Linier')
        st.write(data)

        # Izinkan pengguna memilih kolom tahun (variabel independen) dan jumlah penduduk (variabel dependen)
        x_column = st.selectbox('Pilih kolom Model (Variabel Independen)', data.columns)
        y_column = st.selectbox('Pilih kolom Price (Variabel Dependen)', data.columns)

        # Tampilkan scatter plot untuk melihat hubungan antara tahun dan jumlah penduduk
        plt.figure(figsize=(10, 6))
        plt.scatter(data[x_column], data[y_column])
        plt.title('Scatter Plot: Hubungan antara Model dan Price')
        plt.xlabel('Model')
        plt.ylabel('Price')
        st.pyplot()


# Define a function for the "Perhitungan" page
def Perhitungan_page():
    st.title('Estimasi Harga Mobil Bekas Menggunakan Algoritma Regresi Linier')

    model = pickle.load(open('estimasi1_mobil.sav', 'rb'))

    # Bidang masukan untuk pengguna memasukkan data
    jenis_options = ["Auris", "Avensis", "Aygo", "Camry", "C-HR", "Corolla", "GT86", "Hilux", "IQ", "Land Cruiser", "Prius", "Proace verso", "Rav4", "Supra", "Urban Cruiser", "Verso", "Verso-s", "Yaris"]
    selected_jenis = st.selectbox('Pilih Model', jenis_options)
    year = st.text_input('Input Tahun Mobil')
    transmission_options = ["automatic", "manual", "other", "semi-auto"]
    selected_transmission = st.selectbox('Pilih Transmission', transmission_options)
    mileage = st.text_input('Input km Mobil')
    fuelType_options = ["diesel", "hybrid", "other", "petrol"]
    selected_fuelType = st.selectbox('Pilih Type', fuelType_options)
    tax = st.text_input('Input Pajak Mobil')
    mpg = st.number_input('Input Konsumsi BBM Mobil', value=0.0, step=0.01)
    engineSize = st.text_input('Input Engine Size')

    predict = ''

    if st.button("Dapatkan Estimasi Harga"):
        try:
            jenis_index = jenis_options.index(selected_jenis)
            year = int(year)
            transmission_index = transmission_options.index(selected_transmission)
            mileage = int(mileage)
            fuelType_index = fuelType_options.index(selected_fuelType)
            tax = int(tax)
            mpg = float(mpg)
            engineSize = int(engineSize)
        except ValueError:
            st.warning("Bidang masukan harus berupa bilangan bulat atau desimal. Harap masukkan nilai numerik yang valid.")
            return

        # Make the prediction using the loaded model
        predict = model.predict([[jenis_index, year, transmission_index, mileage, fuelType_index, tax, mpg, engineSize]])

        st.success('Estimasi Harga Mobil Bekas dalam Ponds: {:.2f}'.format(predict[0]))
        st.success('Estimasi Harga Mobil Bekas dalam IDR (Juta): {:.2f}'.format(predict[0] * 19341))

# Call the respective function based on the selected page
if selected == "Dashboard":
    dashboard_page()
elif selected == "Visualisasi Data":
    data_visualization_page()
elif selected == "Perhitungan":
    Perhitungan_page()

