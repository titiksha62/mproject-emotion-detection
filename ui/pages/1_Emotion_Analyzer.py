import streamlit as st
import torch
import torch.nn.functional as F
import tempfile, os
import pandas as pd
import plotly.express as px

from utils.model_loader import load_model, load_face_detector
from utils.preprocessing import process_video, process_audio

EMOTIONS = ['Neutral','Calm','Happy','Sad','Angry','Fearful','Disgust','Surprised']

st.title("🎭 Emotion Analyzer")

model = load_model()
mtcnn = load_face_detector()

col1, col2 = st.columns(2)

video = col1.file_uploader("Upload Video (.mp4)", type=["mp4"])
audio = col2.file_uploader("Upload Audio (.wav)", type=["wav"])

if video and audio:
    st.video(video)
    st.audio(audio)

    if st.button("Analyze Emotion", use_container_width=True):
        with st.spinner("Processing..."):

            # Save temp files
            tv = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tv.write(video.read())
            tv.close()

            ta = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            ta.write(audio.read())
            ta.close()

            # Preprocess
            v = process_video(tv.name, mtcnn)
            a = process_audio(ta.name)

            if v is not None and a is not None:
                v = v.permute(0,2,1,3,4).reshape(-1,3,224,224)

                with torch.no_grad():
                    logits, _, _ = model(a, v)
                    probs = F.softmax(logits, dim=1).numpy().squeeze()

                df = pd.DataFrame({
                    "Emotion": EMOTIONS,
                    "Confidence": probs * 100
                }).sort_values(by="Confidence", ascending=False)

                # SAVE RESULTS FOR OTHER PAGE
                st.session_state["results"] = df

                st.success("✅ Analysis Done! Go to Results Dashboard")

            os.remove(tv.name)
            os.remove(ta.name)