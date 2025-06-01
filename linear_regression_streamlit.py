import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Konfigurasi halaman
st.set_page_config(
    page_title="Food Delivery Analysis",
    page_icon="🍔",
    layout="wide"
)

# Judul aplikasi
st.title('📊 Analisis Waktu Pengiriman Makanan')
st.write("""
Aplikasi ini menganalisis faktor-faktor yang memengaruhi waktu pengiriman makanan berdasarkan dataset Food Delivery Times.
""")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv('Food_Delivery_Times.csv')

df = load_data()

# Tampilkan data
if st.checkbox('Tampilkan Data Mentah'):
    st.subheader('Data Mentah')
    st.write(df)

# Preprocessing data
st.header('🧹 Pra-Pemrosesan Data')
st.write("""
### Penanganan Nilai Hilang
Kolom dengan nilai hilang:
- Weather (30 nilai hilang)
- Traffic_Level (30 nilai hilang)
- Time_of_Day (30 nilai hilang)
- Courier_Experience_yrs (30 nilai hilang)

**Solusi:**
- Kolom kategorikal diisi dengan modus
- Kolom numerik diisi dengan median
""")

# Proses handling missing values
cols_to_fill_mode = ['Weather', 'Traffic_Level', 'Time_of_Day']
for col in cols_to_fill_mode:
    df[col].fillna(df[col].mode()[0], inplace=True)

df['Courier_Experience_yrs'].fillna(df['Courier_Experience_yrs'].median(), inplace=True)

st.success("✅ Penanganan nilai hilang berhasil dilakukan")
st.write("Jumlah nilai hilang setelah penanganan:")
st.write(df.isnull().sum())

# Eksplorasi Data
st.header('🔍 Eksplorasi Data')

# Statistik deskriptif
st.subheader('Statistik Deskriptif')
st.write(df.describe())

# Distribusi waktu pengiriman
st.subheader('Distribusi Waktu Pengiriman')
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(data=df, x='Delivery_Time_min', kde=True, bins=30, ax=ax)
plt.xlabel('Waktu Pengiriman (Menit)')
plt.ylabel('Frekuensi')
plt.title('Distribusi Waktu Pengiriman')
st.pyplot(fig)

st.write("""
**Insight:**
- Distribusi waktu pengiriman cenderung normal
- Waktu pengiriman berkisar antara 8-153 menit
- Rata-rata waktu pengiriman: 56.7 menit
""")

# Analisis Faktor Kategorikal
st.header('📈 Analisis Faktor Kategorikal')

# Pilih faktor untuk dianalisis
factor = st.selectbox(
    'Pilih Faktor Kategorikal:',
    ['Weather', 'Traffic_Level', 'Time_of_Day', 'Vehicle_Type']
)

# Visualisasi pengaruh faktor
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=df, x=factor, y='Delivery_Time_min', ax=ax)
plt.title(f'Pengaruh {factor} pada Waktu Pengiriman')
plt.xticks(rotation=45)
st.pyplot(fig)

# Analisis ANOVA untuk signifikansi
st.subheader('Uji Signifikansi Statistik (ANOVA)')
groups = [group[1]['Delivery_Time_min'].values for group in df.groupby(factor)]
f_val, p_val = stats.f_oneway(*groups)

st.write(f"F-value: {f_val:.4f}")
st.write(f"P-value: {p_val:.4f}")

if p_val < 0.05:
    st.success("✅ Terdapat perbedaan signifikan antar grup")
else:
    st.warning("❌ Tidak terdapat perbedaan signifikan antar grup")

# Analisis Korelasi
st.header('🔗 Analisis Korelasi Numerik')

# Heatmap korelasi
st.subheader('Heatmap Korelasi')
corr = df.corr(numeric_only=True)
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
plt.title('Korelasi Antar Variabel Numerik')
st.pyplot(fig)

st.write("""
**Insight:**
- Korelasi terkuat: Distance_km dan Delivery_Time_min (0.89)
- Preparation_Time_min juga berkorelasi positif dengan Delivery_Time_min (0.53)
- Courier_Experience_yrs berkorelasi negatif dengan Delivery_Time_min (-0.41)
""")

# Analisis Regresi
st.header('📈 Prediksi Waktu Pengiriman')

# Pilih variabel numerik
numerical_var = st.selectbox(
    'Pilih Variabel Numerik untuk Analisis Regresi:',
    ['Distance_km', 'Preparation_Time_min', 'Courier_Experience_yrs']
)

# Visualisasi regresi
fig, ax = plt.subplots(figsize=(10, 6))
sns.regplot(
    data=df, 
    x=numerical_var, 
    y='Delivery_Time_min', 
    scatter_kws={'alpha':0.5},
    line_kws={'color':'red'},
    ax=ax
)
plt.title(f'Hubungan {numerical_var} dan Waktu Pengiriman')
st.pyplot(fig)

# Interpretasi
if numerical_var == 'Distance_km':
    st.write("""
    **Insight:**
    - Hubungan positif yang sangat kuat (korelasi 0.89)
    - Setiap penambahan 1 km jarak meningkatkan waktu pengiriman sekitar 4-5 menit
    """)
elif numerical_var == 'Preparation_Time_min':
    st.write("""
    **Insight:**
    - Hubungan positif sedang (korelasi 0.53)
    - Waktu persiapan yang lebih lama berkontribusi pada waktu pengiriman total
    """)
else:
    st.write("""
    **Insight:**
    - Hubungan negatif (korelasi -0.41)
    - Pengalaman kurier mengurangi waktu pengiriman
    - Setiap tahun pengalaman mengurangi waktu pengiriman sekitar 2-3 menit
    """)

# Kesimpulan
st.header('🎯 Kesimpulan dan Rekomendasi')
st.write("""
1. **Faktor Dominan:** Jarak tempuh adalah faktor terpenting yang memengaruhi waktu pengiriman
2. **Kondisi Cuaca:** 
   - Cuaca Snowy dan Rainy meningkatkan waktu pengiriman signifikan
   - Rekomendasi: Alokasi kurier tambahan saat kondisi cuaca buruk
3. **Tingkat Lalu Lintas:**
   - Lalu lintas High meningkatkan waktu pengiriman 15-20 menit
   - Rekomendasi: Gunakan rute alternatif dan optimasi algoritma routing
4. **Jenis Kendaraan:**
   - Motor lebih cepat 10-15 menit dibanding sepeda
   - Rekomendasi: Prioritaskan penggunaan motor untuk pesanan jarak jauh
5. **Pengalaman Kurier:**
   - Pengalaman mengurangi waktu pengiriman
   - Rekomendasi: Program pelatihan untuk kurier baru
6. **Waktu Operasional:**
   - Pengiriman malam lebih cepat daripada sore/siang
   - Rekomendasi: Pertimbangkan shift tambahan di malam hari
""")

# Tampilkan data yang sudah diproses
st.download_button(
    label="Download Data yang Sudah Diproses",
    data=df.to_csv(index=False).encode('utf-8'),
    file_name='processed_food_delivery_data.csv',
    mime='text/csv'
)
