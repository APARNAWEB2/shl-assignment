import streamlit as st
import librosa
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor

# Load trained model
model = pickle.load(open('model.pkl', 'rb'))  # Save your model using pickle first

# Feature extraction
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs, axis=1)

st.title("Grammar Score Predictor from Audio")

uploaded_file = st.file_uploader("Upload a .wav audio file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    
    # Save temp file
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())
    
    # Extract features and predict
    features = extract_features("temp.wav").reshape(1, -1)
    prediction = model.predict(features)[0]
    
    st.success(f"Predicted Grammar Score: {prediction:.2f}")
