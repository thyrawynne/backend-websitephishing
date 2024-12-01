import streamlit as st
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch import nn

# Inisialisasi model dan scaler
class PhishingModel(nn.Module):
    def __init__(self):
        super(PhishingModel, self).__init__()
        self.fc1 = nn.Linear(87, 64)  # Sesuaikan dengan jumlah fitur
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x

# Muat model
model = PhishingModel()
model.load_state_dict(torch.load('phishing_model.pth', map_location=torch.device('cpu')))
model.eval()

# Dummy scaler untuk normalisasi
scaler = MinMaxScaler()

# Streamlit app
st.title('Phishing Website Detection')
st.write("Input website features for prediction:")

# Input fitur pengguna
features = []
for i in range(87):  # Ganti dengan jumlah fitur yang sesuai
    feature_value = st.number_input(f'Feature {i+1}', min_value=-10, max_value=10)  # Sesuaikan dengan nilai input
    features.append(feature_value)

# Prediksi jika tombol ditekan
if st.button('Predict'):
    try:
        # Normalisasi dan prediksi
        features_array = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features_array)
        features_tensor = torch.FloatTensor(features_scaled)

        with torch.no_grad():
            output = model(features_tensor).item()
            prediction = torch.sigmoid(torch.tensor(output)) > 0.5

        result = "Phishing" if prediction else "Legitimate"
        confidence = float(output)

        st.write(f"Prediction: {result}")
        st.write(f"Confidence: {confidence}")
    except Exception as e:
        st.write(f"Error: {str(e)}")
