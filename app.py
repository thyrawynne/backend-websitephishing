import streamlit as st
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch import nn

# Inisialisasi model dan scaler
class PhishingModel(nn.Module):
    def __init__(self):
        super(PhishingModel, self).__init__()
        self.fc1 = nn.Linear(2, 64)  # Ubah input menjadi 2 fitur
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x

# Muat model
try:
    model = PhishingModel()
    model.load_state_dict(torch.load('phishing_model.pth', map_location=torch.device('cpu')))
    model.eval()
except FileNotFoundError:
    st.error("Model file 'phishing_model.pth' tidak ditemukan. Harap unggah model yang benar.")

# Inisialisasi scaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Streamlit app
st.title('Phishing Website Detection')
st.write("Masukkan fitur website untuk prediksi:")

# Input fitur pengguna (dua fitur saja)
features = []
col1, col2 = st.columns(2)

with col1:
    feature_1 = st.number_input('Feature 1', min_value=-10.0, max_value=10.0, value=0.0)
    features.append(feature_1)
with col2:
    feature_2 = st.number_input('Feature 2', min_value=-10.0, max_value=10.0, value=0.0)
    features.append(feature_2)

# Prediksi jika tombol ditekan
if st.button('Prediksi'):
    try:
        features_array = np.array(features).reshape(1, -1)

        # Normalisasi hanya jika scaler cocok dengan jumlah fitur
        if features_array.shape[1] == 2:
            features_scaled = scaler.fit_transform(features_array)  # Biasanya fit digunakan sebelumnya di training
            features_tensor = torch.FloatTensor(features_scaled)

            with torch.no_grad():
                output = model(features_tensor).item()
                prediction = torch.sigmoid(torch.tensor(output)) > 0.5

            result = "Phishing" if prediction else "Legitimate"
            confidence = torch.sigmoid(torch.tensor(output)).item()

            st.success(f"Prediksi: {result}")
            st.info(f"Confidence: {confidence:.4f}")
        else:
            st.error("Jumlah fitur tidak sesuai.")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {str(e)}")