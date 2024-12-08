import streamlit as st
import joblib

# Memuat model dan vectorizer
model = joblib.load("random_forest_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

st.title("Website Phishing Detection")
st.write("Masukkan URL untuk memprediksi apakah itu phishing, benign, malware, atau defacement.")

# Input pengguna
input_url = st.text_input("Masukkan URL:", "")

if st.button("Prediksi"):
    if input_url:
        # Transformasi URL
        url_vectorized = vectorizer.transform([input_url]).toarray()
        
        # Prediksi kelas
        prediction = model.predict(url_vectorized)
        predicted_label = prediction[0]
        
        # Tampilkan hasil
        st.write(f"Hasil prediksi: *{predicted_label}*")
    else:
        st.error("Harap masukkan URL untuk diprediksi!")
