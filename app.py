import streamlit as st
import numpy as np
import librosa
import tsfel
import joblib
import soundfile as sf
import tempfile
import os
from io import BytesIO
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(
    page_title="Klasifikasi Suara Buka/Tutup",
    page_icon="🎙️",
    layout="wide"
)

# Load model dan artifacts
@st.cache_resource
def load_model_artifacts():
    model = joblib.load('voice_classifier_model.pkl')
    scaler = joblib.load('voice_scaler.pkl')
    feature_names = joblib.load('feature_names.pkl')
    cfg = joblib.load('tsfel_config.pkl')
    metadata = joblib.load('model_metadata.pkl')
    return model, scaler, feature_names, cfg, metadata

# Fungsi preprocessing
def preprocess_audio(y, sr, target_sr=16000):
    """Preprocess audio: resample, trim, normalize"""
    # Resample jika perlu
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    
    # Trim silence
    y_trimmed, _ = librosa.effects.trim(y, top_db=25)
    
    # Normalize
    if np.max(np.abs(y_trimmed)) > 0:
        y_trimmed = y_trimmed / np.max(np.abs(y_trimmed))
    
    return y_trimmed, sr

# Fungsi ekstraksi fitur
def extract_features(y, sr, cfg, feature_names):
    """Ekstrak fitur TSFEL dari audio"""
    try:
        X = tsfel.time_series_features_extractor(cfg, y, fs=sr, verbose=0)
        X_clean = X.replace([np.inf, -np.inf], np.nan)
        
        # Ambil hanya fitur yang digunakan saat training
        features = []
        for fname in feature_names:
            if fname in X_clean.columns:
                features.append(X_clean[fname].values[0])
            else:
                features.append(np.nan)
        
        return np.array(features).reshape(1, -1)
    except Exception as e:
        st.error(f"Error extracting features: {str(e)}")
        return None

# Fungsi prediksi
def predict_audio(y, sr, model, scaler, cfg, feature_names, metadata):
    """Prediksi kelas audio"""
    # Preprocess
    y_proc, sr_proc = preprocess_audio(y, sr, metadata['target_sr'])
    
    # Extract features
    features = extract_features(y_proc, sr_proc, cfg, feature_names)
    
    if features is None:
        return None, None, None
    
    # Check for NaN
    if np.isnan(features).any():
        st.warning("Some features are NaN. Using mean imputation.")
        features = np.nan_to_num(features, nan=0.0)
    
    # Scale and predict
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]
    
    return prediction, probabilities, y_proc

# Fungsi visualisasi
def plot_waveform_and_spectrogram(y, sr):
    """Plot waveform dan spectrogram"""
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    
    # Waveform
    librosa.display.waveshow(y, sr=sr, ax=axes[0])
    axes[0].set_title('Waveform')
    axes[0].set_xlabel('Waktu (detik)')
    axes[0].set_ylabel('Amplitudo')
    axes[0].grid(True, alpha=0.3)
    
    # Spectrogram
    S = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=1024, hop_length=256)), ref=np.max)
    img = librosa.display.specshow(S, sr=sr, hop_length=256, x_axis='time', y_axis='linear', ax=axes[1])
    axes[1].set_title('Spectrogram')
    fig.colorbar(img, ax=axes[1], format='%+2.0f dB')
    
    plt.tight_layout()
    return fig

# Main app
def main():
    # Header
    st.title("🎙️ Klasifikasi Suara Buka/Tutup")
    st.markdown("""
    Aplikasi ini mengklasifikasikan suara menjadi **Buka** atau **Tutup** menggunakan machine learning.
    Anda dapat mengupload file audio atau merekam suara secara langsung.
    """)
    
    # Load model
    with st.spinner('Loading model...'):
        model, scaler, feature_names, cfg, metadata = load_model_artifacts()
    
    # Sidebar - Info Model
    st.sidebar.header("📊 Informasi Model")
    st.sidebar.write(f"**Akurasi Training:** {metadata['train_accuracy']:.2%}")
    st.sidebar.write(f"**Akurasi Testing:** {metadata['test_accuracy']:.2%}")
    st.sidebar.write(f"**Jumlah Fitur:** {metadata['n_features']}")
    st.sidebar.write(f"**Sample Rate:** {metadata['target_sr']} Hz")
    st.sidebar.markdown("---")
    st.sidebar.header("ℹ️ Petunjuk Penggunaan")
    st.sidebar.markdown("""
    1. Pilih metode input (Upload atau Rekam)
    2. Upload file audio (.wav, .mp3) atau rekam suara
    3. Klik tombol **Prediksi**
    4. Lihat hasil prediksi dan visualisasi
    """)
    
    # Main content
    tab1, tab2 = st.tabs(["📁 Upload File Audio", "🎤 Rekam Suara"])
    
    # Tab 1: Upload File
    with tab1:
        st.header("Upload File Audio")
        uploaded_file = st.file_uploader(
            "Pilih file audio (WAV atau MP3)",
            type=['wav', 'mp3'],
            help="Upload file audio dengan format .wav atau .mp3"
        )
        
        if uploaded_file is not None:
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
            
            try:
                # Load audio
                y, sr = librosa.load(tmp_path, sr=None, mono=True)
                
                # Display audio player
                st.audio(uploaded_file, format=f'audio/{uploaded_file.name.split(".")[-1]}')
                
                col1, col2, col3 = st.columns([1, 1, 2])
                with col1:
                    st.metric("Durasi", f"{len(y)/sr:.2f} detik")
                with col2:
                    st.metric("Sample Rate", f"{sr} Hz")
                
                # Predict button
                if st.button("🔍 Prediksi", key="predict_upload", type="primary"):
                    with st.spinner('Memproses audio...'):
                        prediction, probabilities, y_proc = predict_audio(
                            y, sr, model, scaler, cfg, feature_names, metadata
                        )
                        
                        if prediction is not None:
                            # Display results
                            st.success("✅ Prediksi selesai!")
                            
                            # Prediction result
                            result_label = metadata['label_map'][prediction]
                            confidence = probabilities[prediction] * 100
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("### 🎯 Hasil Prediksi")
                                st.markdown(f"## **{result_label}**")
                                st.markdown(f"Confidence: **{confidence:.1f}%**")
                                
                                # Probability bars
                                st.markdown("#### Probabilitas:")
                                for i, label in metadata['label_map'].items():
                                    prob = probabilities[i] * 100
                                    st.progress(probabilities[i])
                                    st.write(f"{label}: {prob:.1f}%")
                            
                            with col2:
                                # Visualizations
                                st.markdown("### 📊 Visualisasi Audio")
                                fig = plot_waveform_and_spectrogram(y_proc, metadata['target_sr'])
                                st.pyplot(fig)
                                plt.close()
                
            finally:
                # Clean up temp file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
    
    # Tab 2: Record Audio
    with tab2:
        st.header("Rekam Suara Langsung")
        st.info("💡 **Tips:** Pastikan mikrofon Anda aktif dan izinkan akses browser ke mikrofon.")
        
        try:
            from audio_recorder_streamlit import audio_recorder
            
            # Audio recorder
            audio_bytes = audio_recorder(
                text="Klik untuk mulai/berhenti merekam",
                recording_color="#e74c3c",
                neutral_color="#3498db",
                icon_name="microphone",
                icon_size="3x"
            )
            
            if audio_bytes:
                # Save to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    tmp_file.write(audio_bytes)
                    tmp_path = tmp_file.name
                
                try:
                    # Load audio
                    y, sr = librosa.load(tmp_path, sr=None, mono=True)
                    
                    # Display audio player
                    st.audio(audio_bytes, format='audio/wav')
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Durasi", f"{len(y)/sr:.2f} detik")
                    with col2:
                        st.metric("Sample Rate", f"{sr} Hz")
                    
                    # Predict button
                    if st.button("🔍 Prediksi", key="predict_record", type="primary"):
                        with st.spinner('Memproses audio...'):
                            prediction, probabilities, y_proc = predict_audio(
                                y, sr, model, scaler, cfg, feature_names, metadata
                            )
                            
                            if prediction is not None:
                                # Display results
                                st.success("✅ Prediksi selesai!")
                                
                                # Prediction result
                                result_label = metadata['label_map'][prediction]
                                confidence = probabilities[prediction] * 100
                                
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown("### 🎯 Hasil Prediksi")
                                    st.markdown(f"## **{result_label}**")
                                    st.markdown(f"Confidence: **{confidence:.1f}%**")
                                    
                                    # Probability bars
                                    st.markdown("#### Probabilitas:")
                                    for i, label in metadata['label_map'].items():
                                        prob = probabilities[i] * 100
                                        st.progress(probabilities[i])
                                        st.write(f"{label}: {prob:.1f}%")
                                
                                with col2:
                                    # Visualizations
                                    st.markdown("### 📊 Visualisasi Audio")
                                    fig = plot_waveform_and_spectrogram(y_proc, metadata['target_sr'])
                                    st.pyplot(fig)
                                    plt.close()
                    
                finally:
                    # Clean up temp file
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
        
        except ImportError:
            st.warning("⚠️ Package 'audio-recorder-streamlit' belum terinstall.")
            st.code("pip install audio-recorder-streamlit", language="bash")
            st.info("Sebagai alternatif, Anda dapat menggunakan fitur Upload File Audio.")

if __name__ == "__main__":
    main()
