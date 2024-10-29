import streamlit as st
import pandas as pd

# Judul aplikasi
st.title("Aplikasi Unduh Data")

# Membuat DataFrame contoh
data = {
    'Tanggal': pd.date_range(start='1/1/2022', periods=10),
    'Harga Penutupan': [150, 152, 153, 148, 155, 157, 160, 162, 164, 167]
}
df = pd.DataFrame(data)

# Tampilkan DataFrame di aplikasi
st.write("Data Harga Penutupan:")
st.dataframe(df)

# Tombol untuk mengunduh CSV
csv = df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Unduh Data sebagai CSV",
    data=csv,
    file_name='harga_penutupan.csv',
    mime='text/csv'
)

# Tombol untuk mengunduh Excel
excel = df.to_excel(index=False, engine='openpyxl')
st.download_button(
    label="Unduh Data sebagai Excel",
    data=excel,
    file_name='harga_penutupan.xlsx',
    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
)
