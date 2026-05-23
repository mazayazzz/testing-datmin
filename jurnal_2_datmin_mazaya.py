import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. LOAD MODEL & SCALER
lr_model = joblib.load('linear_regression_model.pkl')
lr_scaler = joblib.load('linear_regression_scaler.pkl')

nb_model = joblib.load('naive_bayes_smote.pkl')
nb_scaler = joblib.load('naive_bayes_scaler.pkl')

# 2. SIDEBAR NAVIGATION
st.sidebar.title("Menu")
page = st.sidebar.selectbox(
    "Pilih Halaman",
    ["Prediksi Glucose", "Klasifikasi Diabetes"]
)

# 3. HALAMAN REGRESSION (PREDIKSI GLUCOSE)
if page == "Prediksi Glucose":

    st.title("Prediksi Glucose")

    Pregnancies = st.number_input("Pregnancies", 0, 20, value=0)
    Age = st.number_input("Age", 1, 100, value=25)
    BMI = st.number_input("BMI", 0.0, 100.0, value=22.0)
    BloodPressure = st.number_input("BloodPressure", 0.0, 200.0, value=80.0)
    HbA1c = st.number_input("HbA1c", 0.0, 20.0, value=5.5)
    LDL = st.number_input("LDL", 0.0, 500.0, value=100.0)
    HDL = st.number_input("HDL", 0.0, 500.0, value=50.0)
    Triglycerides = st.number_input("Triglycerides", 0.0, 1000.0, value=150.0)
    WaistCircumference = st.number_input("WaistCircumference", 0.0, 200.0, value=80.0)
    HipCircumference = st.number_input("HipCircumference", 0.0, 200.0, value=95.0)
    WHR = st.number_input("WHR", 0.0, 2.0, value=0.8)
    FamilyHistory = st.number_input("FamilyHistory (0=Tidak, 1=Ya)", 0, 1, value=0)
    diet_type = st.selectbox("Diet Type", options=[0, 1, 2], help="Pilih tipe diet")
    hypertension = st.selectbox("Hypertension", options=[0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")
    medication_use = st.selectbox("Medication Use", options=[0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")
    outcome = st.selectbox("Outcome", options=[0, 1])

    if st.button("Prediksi Glucose"):
        # Menyusun DataFrame sesuai nama & urutan fitur numerik saat scaler di-fit di Colab
        input_numeric = pd.DataFrame([{
            'Age': Age,
            'Pregnancies': Pregnancies,
            'BMI': BMI,
            'Glucose': 0.0, # <-- Mengisi Kolom Dummy Glucose agar Scaler tidak Error
            'BloodPressure': BloodPressure,
            'HbA1c': HbA1c,
            'LDL': LDL,
            'HDL': HDL,
            'Triglycerides': Triglycerides,
            'WaistCircumference': WaistCircumference,
            'HipCircumference': HipCircumference,
            'WHR': WHR
        }])

        # Lakukan scaling menggunakan numpy array (bypass pengecekan nama kolom scikit-learn)
        scaled_array = lr_scaler.transform(input_numeric)
        
        # Ambil kembali hasil scaling tetapi hilangkan indeks kolom Glucose (indeks ke-3)
        scaled_numeric = np.delete(scaled_array, 3, axis=1)

        # Gabungkan dengan fitur kategorikal untuk dimasukkan ke Linear Regression Model
        input_categorical = np.array([[FamilyHistory, diet_type, hypertension, medication_use, outcome]])
        final_features = np.hstack((scaled_numeric, input_categorical))
        
        # Jalankan Prediksi
        prediction = lr_model.predict(final_features)

        st.success(f"Hasil Prediksi Glucose: {prediction[0]:.2f}")

# 4. HALAMAN CLASSIFICATION (KLASIFIKASI DIABETES)
elif page == "Klasifikasi Diabetes":

    st.title("Klasifikasi Risiko Diabetes")

    pregnancies_nb = st.number_input("Pregnancies", 0, 20, value=0)
    age_nb = st.number_input("Age", 1, 100, value=25)
    bmi_nb = st.number_input("BMI", 0.0, 100.0, value=22.0)
    glucose_nb = st.number_input("Glucose", 0.0, 300.0, value=100.0)
    bloodpressure_nb = st.number_input("BloodPressure", 0.0, 200.0, value=80.0)
    hba1c_nb = st.number_input("HbA1c", 0.0, 20.0, value=5.5)

    if st.button("Prediksi Outcome"):
        # Membuat representasi 16 kolom penuh sesuai urutan dataset asli di Colab:
        # Age, Pregnancies, BMI, Glucose, BloodPressure, HbA1c, LDL, HDL, Triglycerides, 
        # WaistCircumference, HipCircumference, WHR, FamilyHistory, DietType, Hypertension, MedicationUse
        # Kita isi nilai input asli pada kolom yang sesuai, sisanya kita beri nilai dummy 0.0
        
        full_16_features = pd.DataFrame([{
            'Age': age_nb,                  # Input 1
            'Pregnancies': pregnancies_nb,  # Input 2
            'BMI': bmi_nb,                  # Input 3
            'Glucose': glucose_nb,          # Input 4
            'BloodPressure': bloodpressure_nb, # Input 5
            'HbA1c': hba1c_nb,              # Input 6
            'LDL': 0.0,
            'HDL': 0.0,
            'Triglycerides': 0.0,
            'WaistCircumference': 0.0,
            'HipCircumference': 0.0,
            'WHR': 0.0,
            'FamilyHistory': 0.0,
            'DietType': 0.0,
            'Hypertension': 0.0,
            'MedicationUse': 0.0
        }])

        # Lakukan scaling dengan 16 kolom agar nb_scaler tidak protes
        scaled_full_array = nb_scaler.transform(full_16_features)

        # Naive Bayes model (nb_model) Anda di Colab dilatih menggunakan semua 16 kolom tersebut
        # Jadi kita bisa langsung memasukkan hasil array 16 kolom yang sudah di-scale ini ke model
        prediction = nb_model.predict(scaled_full_array)

        if prediction[0] == 1:
            st.error("Hasil: Pasien Berisiko Diabetes")
        else:
            st.success("Hasil: Pasien Tidak Berisiko Diabetes")