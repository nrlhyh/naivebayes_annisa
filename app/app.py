import streamlit as st
import joblib
import numpy as np

# Load model dan label encoder
model = joblib.load('model/model.pkl')
label_encoder = joblib.load('model/label_encoder.pkl')

st.set_page_config(page_title="Klasifikasi Kualitas Air", page_icon="💧")

st.title("Klasifikasi Kelayakan Air untuk Budidaya Ikan")

with st.form("input_form"):
    kecerahan = st.number_input("Kecerahan", step=0.01)
    kekeruhan = st.number_input("Kekeruhan", step=0.01)
    ph = st.number_input("pH (4 - 9)", min_value=4.0, max_value=9.0, step=0.1)
    suhu = st.number_input("Suhu (°C) (15 - 40)", min_value=15.0, max_value=40.0, step=0.1)
    salinitas = st.number_input("Salinitas", step=0.01)
    tss = st.number_input("TSS", step=0.01)
    bod5 = st.number_input("BOD5", step=0.01)
    do = st.number_input("DO", step=0.01)
    ml = st.number_input("M&L", step=0.01)
    coliform = st.number_input("Coliform", step=0.01)
    no3n = st.number_input("NO3N", step=0.01)
    orthophospate = st.number_input("Orthophospate", step=0.01)

    submitted = st.form_submit_button("Prediksi")

if submitted:
    features = np.array([[kecerahan, kekeruhan, ph, suhu, salinitas, tss, bod5, do, ml, coliform, no3n, orthophospate]])
    prediction_encoded = model.predict(features)[0]
    prediction_label = label_encoder.inverse_transform([prediction_encoded])[0]
    st.success(f"Hasil Prediksi: {prediction_label}")
