import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import librosa
import cv2
import tempfile
import os
from facenet_pytorch import MTCNN
from PIL import Image
import pandas as pd
import plotly.express as px

# Import your custom architecture
from models.multimodalcnn import MultiModalCNN

# --- CONFIGURATION ---
MODEL_PATH = "results/RAVDESS_multimodalcnn_15_best0.pth" # Ensure this path is correct!
NUM_CLASSES = 8
SEQ_LENGTH = 15
EMOTIONS = ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised']

# UI Enhancements
EMOJI_MAP = {
    'Neutral': '😐', 'Calm': '😌', 'Happy': '😊', 'Sad': '😢', 
    'Angry': '😠', 'Fearful': '😨', 'Disgust': '🤢', 'Surprised': '😲'
}

COLOR_MAP = {
    'Happy': '#FFD700', 'Sad': '#1F77B4', 'Angry': '#D62728',
    'Fearful': '#9467BD', 'Disgust': '#2CA02C', 'Surprised': '#FF7F0E',
    'Calm': '#17BECF', 'Neutral': '#7F7F7F'
}

st.set_page_config(page_title="HCR-CAF Emotion Recognition", page_icon="🎭", layout="wide")

@st.cache_resource
def load_model():
    """Loads the HCR-CAF model and weights."""
    model = MultiModalCNN(num_classes=NUM_CLASSES, fusion='hcrcaf', seq_length=SEQ_LENGTH, pretr_ef='None')
    
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=False)
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model
    else:
        st.error(f"Model weights not found at {MODEL_PATH}.")
        return None

@st.cache_resource
def load_face_detector():
    """Loads the MTCNN face detector."""
    return MTCNN(image_size=224, margin=20, keep_all=False, post_process=False, device='cpu')

def process_video(video_path, mtcnn):
    """Extracts 15 face frames from the video."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))
    cap.release()
    
    if not frames: return None
    
    indices = np.linspace(0, len(frames) - 1, SEQ_LENGTH, dtype=int)
    sampled_frames = [frames[i] for i in indices]
    
    face_tensors = []
    for img in sampled_frames:
        face = mtcnn(img)
        if face is not None:
            face_tensors.append(face)
        else:
            face_tensors.append(torch.zeros(3, 224, 224))
            
    # Unchanged mathematical scaling
    video_tensor = torch.stack(face_tensors, dim=0).permute(1, 0, 2, 3) / 255.0
    return video_tensor.unsqueeze(0) 

def process_audio(audio_path):
    """Extracts 10 MFCCs from the dedicated .wav file."""
    try:
        y, sr = librosa.load(audio_path, sr=22050)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10)
        audio_tensor = torch.tensor(mfcc, dtype=torch.float32)
        return audio_tensor.unsqueeze(0) 
    except Exception as e:
        st.error(f"Error extracting audio: {e}")
        return None

# --- UI LAYOUT ---
st.title("Multimodal Emotion Recognition 🎭")
st.markdown("**Architecture:** Hierarchical Contrastive Residual Cross-Attention Fusion (HCR-CAF)")
st.divider()

model = load_model()
mtcnn = load_face_detector()

col_vid, col_aud = st.columns(2)

with col_vid:
    st.markdown("#### 🎞️ 1. Upload Original Video (.mp4)")
    uploaded_video = st.file_uploader("", type=["mp4"], label_visibility="collapsed")
with col_aud:
    st.markdown("#### 🎙️ 2. Upload Matching Audio (.wav)")
    uploaded_audio = st.file_uploader("", type=["wav"], label_visibility="collapsed")

if uploaded_video and uploaded_audio and model:
    st.divider()
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.video(uploaded_video)
        st.audio(uploaded_audio)
        analyze_button = st.button("🧠 Analyze Multimodal Inputs", type="primary", use_container_width=True)
    
    with col2:
        if analyze_button:
            with st.spinner("Extracting features and running HCR-CAF fusion..."):
                
                # File handling (Unchanged)
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tv:
                    tv.write(uploaded_video.read())
                    temp_vid_path = tv.name
                    
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as ta:
                    ta.write(uploaded_audio.read())
                    temp_aud_path = ta.name
                
                # Preprocessing (Unchanged)
                visual_inputs = process_video(temp_vid_path, mtcnn)
                audio_inputs = process_audio(temp_aud_path)
                
                if visual_inputs is not None and audio_inputs is not None:
                    
                    # Tensor Folding (Unchanged)
                    visual_inputs = visual_inputs.permute(0, 2, 1, 3, 4)
                    visual_inputs = visual_inputs.reshape(-1, 3, 224, 224)
                    
                    # Inference (Unchanged)
                    with torch.no_grad():
                        logits, _, _ = model(audio_inputs, visual_inputs)
                        probabilities = F.softmax(logits, dim=1).squeeze().numpy()
                        
                    # --- NEW BEAUTIFIED RESULTS SECTION ---
                    prob_df = pd.DataFrame({
                        'Emotion': EMOTIONS,
                        'Confidence (%)': probabilities * 100
                    }).sort_values(by='Confidence (%)', ascending=False)
                    
                    top_emotion = prob_df.iloc[0]['Emotion']
                    top_conf = prob_df.iloc[0]['Confidence (%)']
                    
                    # Main Result Banner
                    st.success(f"### Predicted Emotion: {top_emotion} {EMOJI_MAP[top_emotion]} ({top_conf:.1f}%)")
                    
                    # Dashboard Layout for Details
                    dash_col1, dash_col2 = st.columns([1, 1.5])
                    
                    with dash_col1:
                        st.markdown("#### 🏆 Top 5 Contenders (Prec@5)")
                        top_5_df = prob_df.head(5)
                        for index, row in top_5_df.iterrows():
                            emo = row['Emotion']
                            conf = row['Confidence (%)']
                            st.markdown(f"**{emo}** {EMOJI_MAP[emo]} - `{conf:.1f}%`")
                            st.progress(conf / 100.0)
                            
                    with dash_col2:
                        st.markdown("#### 📊 Confidence Distribution")
                        fig = px.bar(
                            prob_df, 
                            x='Confidence (%)', 
                            y='Emotion', 
                            orientation='h',
                            color='Emotion',
                            color_discrete_map=COLOR_MAP,
                            text=prob_df['Confidence (%)'].apply(lambda x: f'{x:.1f}%')
                        )
                        fig.update_layout(
                            xaxis_title="",
                            yaxis_title="",
                            showlegend=False,
                            height=350,
                            margin=dict(l=0, r=0, t=10, b=0),
                            yaxis={'categoryorder':'total ascending'},
                            xaxis=dict(showgrid=True, gridcolor='rgba(255, 255, 255, 0.1)')
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                # Safe Windows Cleanup
                try: os.remove(temp_vid_path)
                except PermissionError: pass
                try: os.remove(temp_aud_path)
                except PermissionError: pass