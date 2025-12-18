import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import time

# --- 1. Konfigurasi Halaman (Page Configuration) ---
st.set_page_config(
    page_title="Prediksi Jenis Ikan 🐟 - Analisis Komprehensif",
    page_icon="🎣",
    layout="wide" # Menggunakan layout wide untuk tampilan tab yang lebih luas
)

# --- 2. Fungsi Utama untuk Memuat dan Melatih Model ---
FILE_PATH = "Fish.csv"

@st.cache_data
def load_data(file_path):
    """Memuat data CSV."""
    df = pd.read_csv(file_path)
    # Menghapus baris dengan nilai 'Weight' nol
    df = df[df['Weight'] > 0].reset_index(drop=True)
    return df

@st.cache_resource
def train_model(df):
    """Melatih Model Klasifikasi."""
    
    df_clean = df.copy() # Bekerja dengan salinan data bersih
    
    # 1. Encode Target Variabel (Species)
    le = LabelEncoder()
    df_clean['Species_Encoded'] = le.fit_transform(df_clean['Species'])
    
    # 2. Definisikan Fitur (X) dan Target (y)
    features = ['Weight', 'Length1', 'Length2', 'Length3', 'Height', 'Width']
    X = df_clean[features]
    y = df_clean['Species_Encoded']
    
    # 3. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 4. Latih Model (RandomForestClassifier)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 5. Evaluasi
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True, zero_division=0)
    
    return model, le, features, accuracy, report, X_test, y_test, y_pred

# --- 3. Memuat Data dan Melatih Model ---
try:
    fish_df = load_data(FILE_PATH)
    model, label_encoder, feature_names, model_accuracy, report, X_test, y_test, y_pred = train_model(fish_df)
    species_list = list(label_encoder.classes_)
    
except FileNotFoundError:
    st.error(f"⚠️ **ERROR:** File '{FILE_PATH}' tidak ditemukan. Pastikan file tersebut ada di direktori yang sama.")
    st.stop()
except Exception as e:
    st.error(f"⚠️ **ERROR:** Terjadi kesalahan saat memproses data atau melatih model: {e}")
    st.stop()
    
# --- 4. Desain Sidebar dan Input Prediksi ---
with st.sidebar:
    st.markdown("## ⚙️ Pusat Kontrol Prediksi 🐠")
    
    st.markdown("---") 

    st.markdown("### 🎣 Masukkan Karakteristik Ikan")
    
    # Menampilkan akurasi model
    st.info(f"Akurasi Model (RandomForest): **{model_accuracy:.2f}**")

    # Input Fitur
    weight = st.slider(f"**1. {feature_names[0]} (gram)**", min_value=1.0, max_value=1500.0, value=400.0, step=10.0)
    
    st.markdown("#### Dimensi Panjang (cm)")
    col1, col2 = st.columns(2)
    with col1:
        length1 = st.slider(f"**2. {feature_names[1]}** (Vertikal)", min_value=5.0, max_value=60.0, value=30.0, step=0.1)
    with col2:
        length2 = st.slider(f"**3. {feature_names[2]}** (Diagonal)", min_value=5.0, max_value=60.0, value=33.0, step=0.1)
    
    length3 = st.slider(f"**4. {feature_names[3]}** (Lintang)", min_value=5.0, max_value=60.0, value=35.0, step=0.1)
    
    st.markdown("#### Dimensi Tinggi & Lebar (cm)")
    col3, col4 = st.columns(2)
    with col3:
        height = st.slider(f"**5. {feature_names[4]}** (Tinggi)", min_value=0.1, max_value=30.0, value=10.0, step=0.1)
    with col4:
        width = st.slider(f"**6. {feature_names[5]}** (Lebar)", min_value=0.1, max_value=20.0, value=5.0, step=0.1)
    
    st.markdown("---") 

    # Tombol Prediksi di Sidebar
    predict_button = st.button("🚀 **PREDIKSI JENIS IKAN SEKARANG**", type="primary", use_container_width=True)

    st.markdown("---") 
    st.caption(f"Total Jenis Ikan: **{len(species_list)}**")
    st.write("Dibuat dengan Streamlit & Scikit-learn.")
    
# --- 5. Konten Utama Aplikasi (Main Content) dengan Tabs ---

st.title("Aplikasi Prediksi Jenis Spesies Ikan 🐟📊")
st.markdown("""
Aplikasi komprehensif ini menampilkan data, analisis eksplorasi, model klasifikasi, dan antarmuka prediksi.
""")

tab1, tab2, tab3, tab4 = st.tabs(["🎯 Hasil Prediksi", "📚 Dataset", "🔍 EDA (Exploratory Data Analysis)", "🧠 Modeling & Prediksi"])

# --- TAB 1: Hasil Prediksi (Awalnya kosong, diisi setelah klik tombol) ---
with tab1:
    st.header("🎯 Hasil Prediksi Langsung")
    st.info("Masukkan input di **Sidebar** dan klik tombol **PREDIKSI** untuk melihat hasil di sini!")
    
    # Area untuk menampilkan prediksi
    prediction_placeholder = st.empty()

# --- 6. Logika Prediksi ---
if predict_button:
    
    # Kumpulan fitur input
    input_features = np.array([weight, length1, length2, length3, height, width])
    features_input = input_features.reshape(1, -1)
    
    input_data_df = pd.DataFrame({
        'Fitur': feature_names,
        'Nilai Input': input_features
    })

    with tab1:
        prediction_placeholder.empty() # Bersihkan placeholder
        st.subheader("1️⃣ Data Ikan yang Dimasukkan:")
        st.dataframe(input_data_df, hide_index=True, use_container_width=True)
        
        st.markdown("### 2️⃣ Hasil Prediksi Model:")
        
        with st.spinner('⏳ Model Random Forest sedang memproses data...'):
            time.sleep(1.0) # Simulasikan waktu pemrosesan

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
                
                # Tambahkan fitur yang paling penting (Weight) sebagai penekanan
                important_feature_value = input_features[0] 
                
                col_w, col_other = st.columns([1, 2])
                with col_w:
                     st.metric(label="**Berat Input (gram)**", value=f"{important_feature_value:,.2f}")
                with col_other:
                    st.bar_chart(input_data_df.set_index('Fitur').drop('Weight', axis=0), height=200)

                st.markdown("---")
                st.info(f"**Informasi Jenis:** Jenis ikan **{predicted_species}** adalah salah satu dari {len(species_list)} jenis yang ada dalam dataset pelatihan. Lihat **Tab Modeling** untuk detail akurasi.")

            except Exception as e:
                st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")

# --- TAB 2: Dataset ---
with tab2:
    st.header("📚 Dataset Ikan (`Fish.csv`)")
    st.markdown("Berikut adalah 5 baris pertama dari *dataset* ikan yang digunakan:")
    st.dataframe(fish_df.head(), use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("Statistik Deskriptif")
    st.dataframe(fish_df.describe().T, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("Informasi Dataset")
    
    # Membuat tabel untuk informasi dataset
    info_data = {
        'Kolom': fish_df.columns,
        'Tipe Data': [str(dtype) for dtype in fish_df.dtypes],
        'Missing Values': fish_df.isnull().sum().values,
        'Unique Values': fish_df.nunique().values
    }
    st.dataframe(pd.DataFrame(info_data), hide_index=True, use_container_width=True)

# --- TAB 3: EDA (Exploratory Data Analysis) ---
with tab3:
    st.header("🔍 Analisis Data Eksplorasi (EDA)")
    
    # 1. Distribusi Species
    st.subheader("1. Distribusi Jumlah Jenis Ikan")
    species_counts = fish_df['Species'].value_counts().reset_index()
    species_counts.columns = ['Species', 'Count']
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Count', y='Species', data=species_counts, palette="viridis", ax=ax1)
    ax1.set_title('Distribusi Jumlah Ikan per Jenis')
    ax1.set_xlabel('Jumlah')
    ax1.set_ylabel('Jenis Ikan')
    st.pyplot(fig1)
    
    st.markdown("---")
    
    # 2. Korelasi Antar Fitur
    st.subheader("2. Matriks Korelasi Antar Fitur Numerik")
    numeric_df = fish_df[feature_names]
    corr_matrix = numeric_df.corr()
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax2)
    ax2.set_title('Matriks Korelasi')
    st.pyplot(fig2)
    st.markdown("Korelasi tinggi (mendekati 1) antara dimensi panjang (`Length1`, `Length2`, `Length3`) dan juga dengan `Weight` mengindikasikan bahwa fitur-fitur tersebut sangat terkait dengan ukuran fisik ikan.")
    

    st.markdown("---")
    
    # 3. Box Plot (Weight vs Species)
    st.subheader("3. Perbandingan Berat (Weight) berdasarkan Jenis Ikan")
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    sns.boxplot(x='Species', y='Weight', data=fish_df, ax=ax3, palette='Set2')
    ax3.set_title('Box Plot Berat vs Jenis Ikan')
    ax3.set_xlabel('Jenis Ikan')
    ax3.set_ylabel('Berat (gram)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig3)
    st.markdown("Box plot ini menunjukkan variasi berat yang signifikan di antara jenis-jenis ikan.")

# --- TAB 4: Modeling & Prediksi ---
with tab4:
    st.header("🧠 Modeling & Proses Prediksi")
    
    st.subheader("1. Arsitektur Model: Random Forest")
    st.markdown("""
    Model yang digunakan adalah **Random Forest Classifier** yang merupakan *ensemble method* yang membangun banyak *decision tree* saat pelatihan dan menghasilkan kelas yang merupakan modus dari kelas-kelas (*classification*) yang dikeluarkan oleh masing-masing pohon.
    
    * **Fitur (X):** `Weight`, `Length1`, `Length2`, `Length3`, `Height`, `Width`
    * **Target (y):** `Species` (dikonversi menjadi numerik menggunakan `LabelEncoder`)
    * **Hyperparameter Utama:** `n_estimators=100` (100 pohon dalam hutan)
    """)


    st.subheader("2. Evaluasi Model")
    st.info(f"Akurasi Model pada Data Test (20%): **{model_accuracy:.4f}**")
    
    st.markdown("#### Laporan Klasifikasi (Classification Report)")
    # Konversi dictionary report menjadi DataFrame untuk tampilan yang lebih baik
    report_df = pd.DataFrame(report).transpose().iloc[:-3, :-1] # Hapus baris 'accuracy', 'macro avg', 'weighted avg' dan kolom 'support'
    st.dataframe(report_df.style.highlight_max(axis=0, subset=['precision', 'recall', 'f1-score'], color='lightgreen'), use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("3. Feature Importance (Kepentingan Fitur)")
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='Reds_d', ax=ax4)
    ax4.set_title('Kepentingan Fitur dalam Model Random Forest')
    st.pyplot(fig4)
    st.markdown("Fitur **Weight** terlihat sebagai kontributor utama dalam prediksi jenis ikan.")

    
