import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import os
import asyncio
import pandas as pd
import plotly.express as px
import tempfile
import cv2
import librosa
from facenet_pytorch import MTCNN
from PIL import Image

from models.core.multimodal_tri import MultiModalCNNTri
from src.guardrails.security import guardrail
from src.pipeline.async_queue import AsyncStreamQueue

# --- CONFIGURATION ---
NUM_CLASSES = 8
SEQ_LENGTH = 15
EMOTIONS = ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised']

COLOR_MAP = {
    'Happy': '#FFD700', 'Sad': '#1F77B4', 'Angry': '#D62728',
    'Fearful': '#9467BD', 'Disgust': '#2CA02C', 'Surprised': '#FF7F0E',
    'Calm': '#17BECF', 'Neutral': '#7F7F7F'
}

st.set_page_config(page_title="Tri-Modal HCR-CAF", layout="wide")

st.markdown("""
<style>
    .stApp { background: #0f172a; color: #f8fafc; font-family: 'Inter', sans-serif; }
    .stFileUploader > div { border: 1px dashed rgba(56, 189, 248, 0.4) !important; background-color: rgba(56, 189, 248, 0.05) !important; }
    .stButton>button { background: linear-gradient(90deg, #38bdf8 0%, #8b5cf6 100%); color: white; border: none; font-weight: bold; width: 100%;}
    .result-card { background: rgba(30, 41, 59, 0.9); padding: 20px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.1); }
</style>
""", unsafe_allow_html=True)

st.title("🧠 Tri-Modal HCR-CAF Emotion AI")
st.markdown("End-to-End Multimodal Emotion Recognition (Video + Audio + Text).")

@st.cache_resource
def load_tri_model():
    # Force CPU for stability as requested
    model = MultiModalCNNTri(num_classes=NUM_CLASSES, embed_dim=128, device='cpu')
    # If you have weights, load them here:
    # model.load_state_dict(torch.load("path_to_weights.pth", map_location='cpu'))
    model.eval()
    return model

@st.cache_resource
def load_face_detector():
    return MTCNN(image_size=224, margin=20, keep_all=False, post_process=False, device='cpu')

def extract_visual_frames(video_path, mtcnn):
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
            
    # Shape: (1, 15, 3, 224, 224)
    video_tensor = torch.stack(face_tensors, dim=0).permute(0, 1, 2, 3) / 255.0
    return video_tensor.unsqueeze(0) 

def extract_audio_features(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=22050)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10)
        audio_tensor = torch.tensor(mfcc, dtype=torch.float32)
        # Shape: (1, 10, Time)
        return audio_tensor.unsqueeze(0) 
    except Exception as e:
        st.error(f"Error extracting audio: {e}")
        return None

model = load_tri_model()
mtcnn = load_face_detector()
async_queue = AsyncStreamQueue(max_workers=3)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🎬 Visual")
    uploaded_video = st.file_uploader("Upload MP4 Video", type=["mp4", "flv"], label_visibility="collapsed")

with col2:
    st.markdown("### 🎙️ Acoustic")
    uploaded_audio = st.file_uploader("Upload WAV Audio", type=["wav"], label_visibility="collapsed")
    
with col3:
    st.markdown("### 💬 Lexical")
    text_input = st.text_area("Enter transcript (Optional)", placeholder="Leave blank to use [NO_SPEECH] fallback...")
    
if st.button("🚀 Execute End-to-End Fusion", use_container_width=True):
    if uploaded_video and uploaded_audio:
        with st.spinner("Extracting Faces & MFCCs... Dispatching to Parallel CPU Threads..."):
            
            # 1. Lexical Guardrail & Fallback
            raw_text = text_input.strip() if text_input else "[NO_SPEECH]"
            safe_text = guardrail.sanitize(raw_text)
            st.success(f"**Sanitized Lexical Input:** {safe_text}")
            
            # 2. Save Uploads
            file_ext = os.path.splitext(uploaded_video.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tv:
                tv.write(uploaded_video.read())
                temp_vid_path = tv.name
                
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as ta:
                ta.write(uploaded_audio.read())
                temp_aud_path = ta.name
                
            # 3. True Unimodal Extraction
            visual_tensor = extract_visual_frames(temp_vid_path, mtcnn)
            acoustic_tensor = extract_audio_features(temp_aud_path)
            lexical_list = [safe_text]
            
            if visual_tensor is not None and acoustic_tensor is not None:
                # 4. Async Queue Processing
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                logits = loop.run_until_complete(
                    async_queue.dispatch_parallel(model, lexical_list, visual_tensor, acoustic_tensor)
                )
                loop.close()
                
                # 5. Results Display
                probabilities = F.softmax(logits, dim=1).detach().squeeze().numpy()
                prob_df = pd.DataFrame({'Emotion': EMOTIONS, 'Confidence': probabilities * 100}).sort_values(by='Confidence', ascending=False)
                
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                res1, res2 = st.columns([1, 2])
                
                top_emo = prob_df.iloc[0]['Emotion']
                top_conf = prob_df.iloc[0]['Confidence']
                
                with res1:
                    st.markdown("#### Primary Prediction")
                    st.markdown(f"<h1 style='color: #38bdf8;'>{top_emo}</h1>", unsafe_allow_html=True)
                    st.markdown(f"**Confidence:** `{top_conf:.1f}%`")
                    st.caption("Fused via Tri-Way Cross Attention")
                    
                with res2:
                    fig = px.bar(prob_df, x='Confidence', y='Emotion', orientation='h', color='Emotion', color_discrete_map=COLOR_MAP)
                    fig.update_layout(xaxis_title="Confidence (%)", yaxis_title="", showlegend=False, height=250, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
                    st.plotly_chart(fig, use_container_width=True)
                    
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.error("Failed to extract features from the uploaded files.")
                
            # Cleanup
            try: os.remove(temp_vid_path)
            except: pass
            try: os.remove(temp_aud_path)
            except: pass
    else:
        st.error("Please upload both a Video and an Audio file to run the end-to-end pipeline!")