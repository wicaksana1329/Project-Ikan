import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import time

# --- 1. Konfigurasi Halaman (Page Configuration) ---
st.set_page_config(
    page_title="Prediksi Jenis Ikan 🐟",
    page_icon="🎣",
    layout="centered"
)

# --- 2. Fungsi Utama untuk Memuat dan Melatih Model ---
# Menggunakan cache agar data hanya dimuat dan model hanya dilatih sekali
@st.cache_data
def load_data(file_path):
    """Memuat data CSV."""
    df = pd.read_csv(file_path)
    return df

@st.cache_resource
def train_model(df):
    """Melatih Model Klasifikasi."""
    
    # 1. Pembersihan Data dan Feature Engineering
    # Menghapus baris dengan nilai 'Weight' nol (jika ada, yang mengganggu prediksi)
    df = df[df['Weight'] > 0].reset_index(drop=True)

    # 2. Encode Target Variabel (Species)
    le = LabelEncoder()
    df['Species_Encoded'] = le.fit_transform(df['Species'])
    
    # 3. Definisikan Fitur (X) dan Target (y)
    features = ['Weight', 'Length1', 'Length2', 'Length3', 'Height', 'Width']
    X = df[features]
    y = df['Species_Encoded']
    
    # 4. Split Data (Untuk evaluasi, meskipun di aplikasi ini fokusnya prediksi)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 5. Latih Model (Menggunakan RandomForestClassifier)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 6. Evaluasi (opsional, untuk ditampilkan)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, le, features, accuracy

# --- 3. Memuat Data dan Melatih Model ---
FILE_PATH = "Fish.csv"
try:
    fish_df = load_data(FILE_PATH)
    model, label_encoder, feature_names, model_accuracy = train_model(fish_df)
    species_list = list(label_encoder.classes_)
    
except FileNotFoundError:
    st.error(f"⚠️ **ERROR:** File '{FILE_PATH}' tidak ditemukan. Pastikan file tersebut ada di direktori yang sama.")
    st.stop()
except Exception as e:
    st.error(f"⚠️ **ERROR:** Terjadi kesalahan saat memproses data atau melatih model: {e}")
    st.stop()
    
# --- 4. Desain Sidebar yang Sangat Menarik ---
with st.sidebar:
    st.markdown("## ⚙️ Pusat Kontrol Prediksi 🐠")
    
    st.markdown("---") 

    st.markdown("### 🎣 Masukkan Karakteristik Ikan")
    
    # Menampilkan akurasi model
    st.info(f"Akurasi Model (RandomForest): **{model_accuracy:.2f}**")

    # Input Fitur (sesuai dengan Fish.csv)
    
    # Weight
    weight = st.number_input(
        f"**1. {feature_names[0]} (gram)**", 
        min_value=1.0, 
        max_value=1500.0, 
        value=400.0, 
        step=10.0, 
        help="Berat ikan, harus lebih dari 0."
    )
    
    # Lengths (diubah menjadi slider/input yang lebih user-friendly)
    st.markdown("#### Dimensi Panjang (cm)")
    col1, col2 = st.columns(2)
    with col1:
        length1 = st.slider(f"**2. {feature_names[1]}** (Vertikal)", 
                            min_value=5.0, max_value=60.0, value=30.0, step=0.1)
    with col2:
        length2 = st.slider(f"**3. {feature_names[2]}** (Diagonal)", 
                            min_value=5.0, max_value=60.0, value=33.0, step=0.1)
    
    length3 = st.slider(f"**4. {feature_names[3]}** (Lintang)", 
                        min_value=5.0, max_value=60.0, value=35.0, step=0.1)
    
    # Height and Width
    st.markdown("#### Dimensi Tinggi & Lebar (cm)")
    col3, col4 = st.columns(2)
    with col3:
        height = st.number_input(f"**5. {feature_names[4]}** (Tinggi)", 
                                 min_value=0.1, max_value=30.0, value=10.0, step=0.1)
    with col4:
        width = st.number_input(f"**6. {feature_names[5]}** (Lebar)", 
                                min_value=0.1, max_value=20.0, value=5.0, step=0.1)
    
    st.markdown("---") 

    # Tombol Prediksi di Sidebar
    predict_button = st.button("🚀 **PREDIKSI JENIS IKAN SEKARANG**", type="primary", use_container_width=True)

    st.markdown("---") 
    
    # Footer Sidebar
    st.caption(f"Total Jenis Ikan dalam Data: **{len(species_list)}**")
    st.write("Dibuat dengan Streamlit & Scikit-learn.")
    
# --- 5. Konten Utama Aplikasi (Main Content) ---

st.title("Aplikasi Prediksi Jenis Ikan 🐟📊")
st.markdown("""
Aplikasi ini menggunakan Model **Random Forest** yang dilatih pada data `Fish.csv` Anda. 
Masukkan karakteristik fisik ikan (Berat, Panjang, Tinggi, Lebar) di **Sidebar** 👈 untuk mendapatkan klasifikasi jenis ikan yang paling mungkin.
""")

# Menampilkan data input yang dimasukkan
input_features = np.array([weight, length1, length2, length3, height, width])
input_data_df = pd.DataFrame({
    'Fitur': feature_names,
    'Nilai Input': input_features
})

st.subheader("1️⃣ Data Ikan yang Dimasukkan:")
st.dataframe(input_data_df, hide_index=True, use_container_width=True)

# --- 6. Logika Prediksi ---

if predict_button:
    
    # Kumpulan fitur input (harus di-reshape untuk model)
    features_input = input_features.reshape(1, -1)
    
    st.markdown("### 2️⃣ Hasil Prediksi Model:")
    
    with st.spinner('⏳ Model Random Forest sedang memproses data...'):
        time.sleep(1.5) # Simulasikan waktu pemrosesan

        try:
            # Lakukan Prediksi
            prediction_encoded = model.predict(features_input)[0]
            
            # Decode Prediksi
            predicted_species = label_encoder.inverse_transform([prediction_encoded])[0]

            # Tampilkan hasil
            st.success(f"✅ Prediksi Selesai!")
            st.snow() 

            st.markdown(f"""
            <div style='background-color: #e6f7ff; padding: 25px; border-radius: 12px; border: 2px solid #007bff;'>
                <h3 style='color: #007bff; margin-top: 0;'>Jenis Ikan yang Diprediksi Adalah:</h3>
                <center>
                    <h1>**{predicted_species}**</h1>
                </center>
                <p><i>Model mengklasifikasikan ikan ini berdasarkan 6 dimensi fisik yang Anda masukkan.</i></p>
            </div>
            """, unsafe_allow_html=True)

            # Tambahkan visualisasi fitur input
            st.subheader("3️⃣ Visualisasi Data Input Relatif")
            st.bar_chart(input_data_df.set_index('Fitur'))
            
            # Tampilkan informasi singkat tentang jenis ikan yang diprediksi
            st.markdown("---")
            st.info(f"**Informasi Jenis:** Jenis ikan **{predicted_species}** adalah salah satu dari {len(species_list)} jenis yang ada dalam dataset pelatihan.")
            # Diagram Anatomi Ikan
            st.markdown("Untuk membantu pemahaman fitur, di bawah adalah diagram umum anatomi ikan:")
            


        except Exception as e:
            st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")

elif not predict_button:
    st.info("💡 Klik tombol **🚀 PREDIKSI JENIS IKAN SEKARANG** di *sidebar* untuk melihat hasil klasifikasi model!")

st.markdown("---")
st.caption("Aplikasi ini dibuat secara dinamis menggunakan Streamlit dan Model Random Forest yang dilatih dari `Fish.csv`.")