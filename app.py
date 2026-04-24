import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import os
import asyncio
import pandas as pd
import plotly.express as px

from models.core.multimodal_tri import MultiModalCNNTri
from src.guardrails.security import guardrail
from src.pipeline.async_queue import AsyncStreamQueue

# --- CONFIGURATION ---
NUM_CLASSES = 8
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
st.markdown("Visual (YOLOv8/InsightFace) + Acoustic (AST/1D-CNN) + Lexical (RoBERTa) Fusion via Local Async Queue.")

@st.cache_resource
def load_tri_model():
    # Force CPU for stability as requested
    model = MultiModalCNNTri(num_classes=NUM_CLASSES, embed_dim=128, device='cpu')
    model.eval()
    return model

model = load_tri_model()
async_queue = AsyncStreamQueue(max_workers=3)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🎬 Visual")
    st.info("Simulated spatial-temporal frames")
    v_ready = st.checkbox("Attach Visual Stream", value=True)

with col2:
    st.markdown("### 🎙️ Acoustic")
    st.info("Simulated MFCC/AST features")
    a_ready = st.checkbox("Attach Acoustic Stream", value=True)
    
with col3:
    st.markdown("### 💬 Lexical")
    text_input = st.text_area("Enter transcript / spoken text", placeholder="I am so incredibly happy today!")
    
if st.button("🚀 Execute Async Tri-Modal Fusion", use_container_width=True):
    if text_input:
        with st.spinner("Processing through Lexical Guardrails & Dispatching to Parallel CPU Threads..."):
            
            # 1. Guardrail / Sanitization
            safe_text = guardrail.sanitize(text_input)
            st.success(f"**Sanitized Lexical Input:** {safe_text}")
            
            # 2. Mocking Unimodal Data loading (Since we don't have the heavy extractors actively running real files yet)
            # visual shape: (Batch, Seq, C, H, W) -> (1, 15, 3, 224, 224)
            dummy_visual = torch.randn(1, 15, 3, 224, 224)
            # acoustic shape: (Batch, 10, Time) -> (1, 10, 156)
            dummy_acoustic = torch.randn(1, 10, 156)
            lexical_list = [safe_text]
            
            # 3. Async Queue Processing
            # Streamlit is synchronous, so we run the asyncio event loop manually
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            logits = loop.run_until_complete(
                async_queue.dispatch_parallel(model, lexical_list, dummy_visual, dummy_acoustic)
            )
            loop.close()
            
            # 4. Results Display
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
        st.error("Please enter Lexical text to guide the attention fusion!")