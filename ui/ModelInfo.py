import streamlit as st

st.title("🧠 Model Information")

st.markdown("""
### Architecture: HCR-CAF

**Hierarchical Contrastive Residual Cross-Attention Fusion**

### 🔍 Modalities:
- 🎥 Visual (Face frames)
- 🔊 Audio (MFCC features)

### ⚙️ Pipeline:
1. Face detection (MTCNN)
2. Frame sampling (15 frames)
3. MFCC extraction
4. Multimodal fusion

### 📊 Dataset:
- RAVDESS

### 🎯 Classes:
- Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised
""")