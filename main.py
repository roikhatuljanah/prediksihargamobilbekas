import streamlit as st
import pandas as pd
from streamlit-option-menu import option_menu
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
        x_column = st.selectbox('Pilih kolom Tahun (Variabel Independen)', data.columns)
        y_column = st.selectbox('Pilih kolom Jumlah Penduduk (Variabel Dependen)', data.columns)

        # Tampilkan scatter plot untuk melihat hubungan antara tahun dan jumlah penduduk
        plt.figure(figsize=(10, 6))
        plt.scatter(data[x_column], data[y_column])
        plt.title('Scatter Plot: Hubungan antara Tahun dan Jumlah Penduduk')
        plt.xlabel('Tahun')
        plt.ylabel('Jumlah Penduduk')
        st.pyplot()

        # Bangun model regresi linier (gunakan pustaka scikit-learn)
        from sklearn.linear_model import LinearRegression

        st.set_option('deprecation.showPyplotGlobalUse', False)
        # Pisahkan data menjadi data pelatihan dan data pengujian (misalnya, 80% pelatihan, 20% pengujian)
        from sklearn.model_selection import train_test_split
        X = data[[x_column]]
        y = data[y_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        st.set_option('deprecation.showPyplotGlobalUse', False)
        # Latih model regresi linier
        model = LinearRegression()
        model.fit(X_train, y_train)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        # Buat prediksi menggunakan model
        y_pred = model.predict(X_test)

        st.set_option('deprecation.showPyplotGlobalUse', False)
        # Tampilkan grafik hasil prediksi
        plt.figure(figsize=(10, 6))
        plt.scatter(X_test, y_test, color='blue', label='Data Aktual')
        plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regresi Linier')
        plt.title('Regresi Linier untuk Perkiraan Jumlah Penduduk')
        plt.xlabel('Tahun')
        plt.ylabel('Jumlah Penduduk')
        plt.legend()
        st.pyplot()

        df.plot(x="Tahun",y="Jumlah",style="o")
        plt.title("Tahun vs Jumlah")
        plt.xlabel("Tahun")
        plt.ylabel("Jumlah")
        plt.show()
        st.pyplot()

        sns.jointplot(x=df['Tahun'],y=df['Jumlah'],data=df,kind='reg')
        st.pyplot()

        plt.scatter(X_test, y_test)
        plt.plot(X_test, y_test, color='red', linewidth=3)
        plt.xlabel('Tahun')
        plt.ylabel('Jumlah')
        plt.title('Linear Regression')
        st.pyplot()
        

# Define a function for the "Perhitungan" page
def Perhitungan_page():
    st.title('Estimasi Harga Mobil Bekas Menggunakan Algoritma Regresi Linier')

    model = pickle.load(open('estimasi_mobil.sav', 'rb'))

    # Bidang masukan untuk pengguna memasukkan data
    year = st.number_input('Input Tahun Mobil')
    mileage = st.number_input('Input km Mobil')
    tax = st.number_input('Input Pajak Mobil')
    mpg = st.number_input('Input Konsumsi BBM Mobil')
    engineSize = st.number_input('Input Engine Size')

    predict = ''

    if st.button("Dapatkan Estimasi Harga"):
        predict = model.predict(
            [[year, mileage, tax, mpg, engineSize]]
        )
        st.write ('Estimasi Harga Mobil Bekas dalam Ponds : ', predict)
        st.write ('Estimasi Harga Mobil Bekas dalam IDR (Juta) : ', predict*19341)


# Call the respective function based on the selected page
if selected == "Dashboard":
    dashboard_page()
elif selected == "Visualisasi Data":
    data_visualization_page()
elif selected == "Perhitungan":
    Perhitungan_page()

