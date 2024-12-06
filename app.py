import streamlit as st
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Muat model yang telah disimpan
model = torch.load('phishing_model.pth')
model.eval()

# Fungsi untuk memprediksi phishing
def predict(input_data):
    # Skalakan data input
    scaler = MinMaxScaler()
    input_scaled = scaler.fit_transform(np.array(input_data).reshape(1, -1))
    
    # Ubah input ke tensor
    input_tensor = torch.FloatTensor(input_scaled)
    
    # Prediksi menggunakan model
    with torch.no_grad():
        prediction = model(input_tensor)
    
    # Menggunakan threshold 0.5 untuk klasifikasi
    return 'Phishing' if prediction.item() > 0.5 else 'Legitimate'

# Antarmuka pengguna Streamlit
st.title('Phishing Website Detection')

st.write("Masukkan dua fitur untuk prediksi phishing:")

# Input fitur dari pengguna (dua fitur)
feature_1 = st.number_input('Feature 1', min_value=0.0, max_value=1.0, step=0.01)
feature_2 = st.number_input('Feature 2', min_value=0.0, max_value=1.0, step=0.01)

# Mengambil input fitur
input_data = [feature_1, feature_2]

if st.button('Prediksi'):
    result = predict(input_data)
    st.write(f'Website ini {result}!')
