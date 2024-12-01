from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from fastapi.middleware.cors import CORSMiddleware

# Inisialisasi FastAPI
app = FastAPI()

# Middleware CORS untuk mengizinkan permintaan dari GitHub Pages
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Ubah "*" ke domain spesifik jika perlu, misalnya "https://username.github.io"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model Input Data
class InputData(BaseModel):
    features: list

# Definisi Model PyTorch
class PhishingModel(nn.Module):
    def __init__(self):
        super(PhishingModel, self).__init__()
        self.fc1 = nn.Linear(87, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x

# Muat model PyTorch
model = PhishingModel()
model.load_state_dict(torch.load('phishing_model.pth', map_location=torch.device('cpu')))
model.eval()

# Dummy MinMaxScaler (sesuaikan sesuai kebutuhan)
scaler = MinMaxScaler()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Phishing Detection API using FastAPI!"}

@app.post("/predict/")
def predict(data: InputData):
    try:
        features = np.array(data.features).reshape(1, -1)
        features_scaled = scaler.transform(features)
        features_tensor = torch.FloatTensor(features_scaled)

        with torch.no_grad():
            output = model(features_tensor).item()
            prediction = torch.sigmoid(torch.tensor(output)) > 0.5

        result = "phishing" if prediction else "legitimate"
        return {"prediction": result, "confidence": float(output)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))