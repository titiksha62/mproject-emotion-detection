import streamlit as st
from utils.model_loader import load_model, load_face_detector
from utils.preprocessing import process_video, process_audio
import tempfile, os, torch
import pandas as pd
import plotly.express as px

EMOTIONS = ['Neutral','Calm','Happy','Sad','Angry','Fearful','Disgust','Surprised']

st.title("🎭 Emotion Analyzer")

model = load_model()
mtcnn = load_face_detector()

col1, col2 = st.columns(2)

video = col1.file_uploader("Upload Video", type=["mp4"])
audio = col2.file_uploader("Upload Audio", type=["wav"])

if video and audio:
    st.video(video)
    st.audio(audio)

    if st.button("Analyze", use_container_width=True):
        with st.spinner("Analyzing..."):

            # temp save
            tv = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tv.write(video.read())

            ta = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            ta.write(audio.read())

            v = process_video(tv.name, mtcnn)
            a = process_audio(ta.name)

            v = v.permute(0,2,1,3,4).reshape(-1,3,224,224)

            with torch.no_grad():
                logits, _, _ = model(a, v)

            probs = torch.softmax(logits, dim=1).numpy().squeeze()

            df = pd.DataFrame({
                "Emotion": EMOTIONS,
                "Confidence": probs*100
            }).sort_values(by="Confidence", ascending=False)

            st.session_state["results"] = df  # 🔥 store globally

            st.success("Analysis complete! Go to Results Dashboard →")

            os.remove(tv.name)
            os.remove(ta.name)