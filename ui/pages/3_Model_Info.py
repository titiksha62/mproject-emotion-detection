import streamlit as st

st.title("🧠 Model Info")

st.markdown("""
### Architecture: HCR-CAF

Hierarchical Contrastive Residual Cross-Attention Fusion

### Pipeline:
1. Face Detection (MTCNN)
2. Frame Sampling (15 frames)
3. MFCC Extraction
4. Multimodal Fusion

### Dataset:
RAVDESS

### Emotions:
Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised
""")