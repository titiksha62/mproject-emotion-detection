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
import plotly.graph_objects as go

# Import custom architecture
from models.multimodalcnn import MultiModalCNN

# --- CONFIGURATION ---
MODEL_PATH = "results/RAVDESS_multimodalcnn_15_best0.pth"
NUM_CLASSES = 8
SEQ_LENGTH = 15
EMOTIONS = ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised']

# UI Styling Tokens
COLOR_MAP = {
    'Happy': '#FFD700', 'Sad': '#1F77B4', 'Angry': '#D62728',
    'Fearful': '#9467BD', 'Disgust': '#2CA02C', 'Surprised': '#FF7F0E',
    'Calm': '#17BECF', 'Neutral': '#7F7F7F'
}
EMOJI_MAP = {
    'Neutral': '😐', 'Calm': '😌', 'Happy': '😊', 'Sad': '😢', 
    'Angry': '😠', 'Fearful': '😨', 'Disgust': '🤢', 'Surprised': '😲'
}

st.set_page_config(page_title="HCR-CAF Emotion AI", page_icon="🎭", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM CSS FOR PREMIUM LOOK ---
st.markdown("""
<style>
    /* Main Background & Text */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #f8fafc !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }
    
    /* Upload Boxes */
    .stFileUploader > div {
        background-color: rgba(255,255,255,0.05) !important;
        border: 1px dashed rgba(255,255,255,0.2) !important;
        border-radius: 12px;
        padding: 1rem;
        transition: all 0.3s ease;
    }
    .stFileUploader > div:hover {
        border-color: #38bdf8 !important;
        background-color: rgba(56, 189, 248, 0.05) !important;
    }
    
    /* Primary Button */
    .stButton>button {
        background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        transition: transform 0.2s, box-shadow 0.2s;
        width: 100%;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(139, 92, 246, 0.3);
    }
    
    /* Cards for Results */
    .result-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 24px;
        margin-top: 20px;
    }
    
    .metric-value {
        font-size: 3rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #38bdf8, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
</style>
""", unsafe_allow_html=True)

# --- BACKEND LOGIC ---
@st.cache_resource
def load_model():
    model = MultiModalCNN(num_classes=NUM_CLASSES, fusion='hcrcaf', seq_length=SEQ_LENGTH, pretr_ef='None')
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=False)
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model
    else:
        st.error(f"⚠️ Model weights not found at {MODEL_PATH}.")
        return None

@st.cache_resource
def load_face_detector():
    return MTCNN(image_size=224, margin=20, keep_all=False, post_process=False, device='cpu')

def process_video(video_path, mtcnn):
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
            
    video_tensor = torch.stack(face_tensors, dim=0).permute(1, 0, 2, 3) / 255.0
    return video_tensor.unsqueeze(0) 

def process_audio(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=22050)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10)
        audio_tensor = torch.tensor(mfcc, dtype=torch.float32)
        return audio_tensor.unsqueeze(0) 
    except Exception as e:
        st.error(f"Error extracting audio: {e}")
        return None

# --- UI LAYOUT ---
st.title("🎭 Multimodal Emotion Intelligence")
st.markdown("End-to-End fusion of Facial Expressions and Vocal Intonations using Hierarchical Contrastive Residual Cross-Attention (HCR-CAF).")

model = load_model()
mtcnn = load_face_detector()

st.write("---")

col_input1, col_input2 = st.columns(2)

with col_input1:
    st.markdown("### 🎬 Visual Modality")
    uploaded_video = st.file_uploader("Upload MP4 Video", type=["mp4", "flv"], label_visibility="collapsed")
    if uploaded_video:
        st.video(uploaded_video)

with col_input2:
    st.markdown("### 🎙️ Audio Modality")
    uploaded_audio = st.file_uploader("Upload WAV Audio", type=["wav"], label_visibility="collapsed")
    if uploaded_audio:
        st.audio(uploaded_audio)

if uploaded_video and uploaded_audio and model:
    st.write("---")
    
    col_btn, col_blank = st.columns([1, 2])
    with col_btn:
        analyze_button = st.button("🧠 Analyze Fusion Intelligence", use_container_width=True)
    
    if analyze_button:
        with st.spinner("Extracting MTCNN Faces & Librosa Audio... Running HCR-CAF..."):
            file_ext = os.path.splitext(uploaded_video.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tv:
                tv.write(uploaded_video.read())
                temp_vid_path = tv.name
                
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as ta:
                ta.write(uploaded_audio.read())
                temp_aud_path = ta.name
            
            visual_inputs = process_video(temp_vid_path, mtcnn)
            audio_inputs = process_audio(temp_aud_path)
            
            if visual_inputs is not None and audio_inputs is not None:
                # Shape formatting for CNN
                visual_inputs = visual_inputs.permute(0, 2, 1, 3, 4)
                visual_inputs = visual_inputs.reshape(-1, 3, 224, 224)
                
                with torch.no_grad():
                    logits, _, _ = model(audio_inputs, visual_inputs)
                    probabilities = F.softmax(logits, dim=1).squeeze().numpy()
                    
                prob_df = pd.DataFrame({
                    'Emotion': EMOTIONS,
                    'Confidence': probabilities * 100
                }).sort_values(by='Confidence', ascending=False)
                
                top_emotion = prob_df.iloc[0]['Emotion']
                top_conf = prob_df.iloc[0]['Confidence']
                
                # --- RESULTS DASHBOARD ---
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                
                res_col1, res_col2 = st.columns([1, 2])
                
                with res_col1:
                    st.markdown("### 🎯 Primary State")
                    st.markdown(f'<div class="metric-value">{EMOJI_MAP[top_emotion]} {top_emotion}</div>', unsafe_allow_html=True)
                    st.markdown(f"**Confidence:** `{top_conf:.2f}%`")
                    
                    if top_conf < 50.0:
                        st.warning("⚠️ Low confidence detection. Subject may be expressing blended emotions.")
                    
                    # Show MTCNN debug frame
                    st.markdown("#### AI Visual Feed")
                    debug_img = visual_inputs[0, :, :, :].permute(1, 2, 0).numpy()
                    debug_img = np.clip(debug_img, 0.0, 1.0)
                    st.image(debug_img, width=150, caption="First Frame MTCNN Crop")

                with res_col2:
                    st.markdown("### 📊 Distribution Spectrum")
                    
                    fig = px.bar(
                        prob_df, 
                        x='Confidence', 
                        y='Emotion', 
                        orientation='h',
                        color='Emotion',
                        color_discrete_map=COLOR_MAP,
                        text=prob_df['Confidence'].apply(lambda x: f'{x:.1f}%')
                    )
                    
                    fig.update_layout(
                        xaxis_title="Confidence Probability (%)",
                        yaxis_title="",
                        showlegend=False,
                        height=350,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#f8fafc'),
                        yaxis={'categoryorder':'total ascending'},
                        xaxis=dict(showgrid=True, gridcolor='rgba(255, 255, 255, 0.1)', range=[0, 100])
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                st.markdown('</div>', unsafe_allow_html=True)
                
            try: os.remove(temp_vid_path)
            except: pass
            try: os.remove(temp_aud_path)
            except: pass