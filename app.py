import sys
import streamlit as st

st.write("Python version:", sys.version)
import numpy as np
import librosa
import tensorflow as tf

# โหลดโมเดล
model = tf.keras.models.load_model("best_model-5.keras")

st.title("Heart Murmur Detection")
st.write("อัปโหลดไฟล์เสียงหัวใจ (.wav) เพื่อวิเคราะห์")

uploaded_file = st.file_uploader("อัปโหลดไฟล์เสียง", type=["wav"])

def extract_features(audio_data, sr):
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=40)
    mfccs = np.mean(mfccs.T, axis=0)
    return mfccs

if uploaded_file is not None:
    audio, sr = librosa.load(uploaded_file, sr=22050)
    features = extract_features(audio, sr)
    features = np.expand_dims(features, axis=0)

    prediction = model.predict(features)[0][0]

    st.subheader("ผลการวิเคราะห์:")
    if prediction >= 0.5:
        st.error(f"ตรวจพบเสียง Murmur (ความน่าจะเป็น: {prediction:.2f})")
    else:
        st.success(f"ไม่พบเสียง Murmur (ความน่าจะเป็น: {prediction:.2f})")
