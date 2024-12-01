import streamlit as st
import torch
import numpy as np
from torch import nn

# Inisialisasi model sederhana untuk 2 fitur
class PhishingModel(nn.Module):
    def __init__(self):
        super(PhishingModel, self).__init__()
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
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

# Streamlit app
st.title('Phishing Website Detection')
st.write("Masukkan fitur website untuk prediksi:")

# Input untuk 2 fitur
feature_1 = st.number_input('Feature 1', min_value=-10.0, max_value=10.0, value=0.0)
feature_2 = st.number_input('Feature 2', min_value=-10.0, max_value=10.0, value=0.0)

# Prediksi jika tombol ditekan
if st.button('Prediksi'):
    try:
        features_array = np.array([[feature_1, feature_2]])
        features_tensor = torch.FloatTensor(features_array)

        with torch.no_grad():
            output = model(features_tensor).item()
            prediction = torch.sigmoid(torch.tensor(output)) > 0.5

        result = "Phishing" if prediction else "Legitimate"
        confidence = torch.sigmoid(torch.tensor(output)).item()

        st.success(f"Prediksi: {result}")
        st.info(f"Confidence: {confidence:.4f}")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {str(e)}")
