# # import streamlit as st
# # import torch
# # import torch.nn.functional as F
# # import torchvision.transforms as transforms
# # import numpy as np
# # import librosa
# # import cv2
# # import tempfile
# # import os
# # from facenet_pytorch import MTCNN
# # from PIL import Image
# # import pandas as pd

# # # Import your custom architecture
# # from models.multimodalcnn import MultiModalCNN

# # # --- CONFIGURATION ---
# # MODEL_PATH = "results/RAVDESS_multimodalcnn_15_best0.pth" # Ensure this path is correct!
# # NUM_CLASSES = 8
# # SEQ_LENGTH = 15
# # EMOTIONS = ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised']

# # st.set_page_config(page_title="HCR-CAF Emotion Recognition", layout="wide")

# # @st.cache_resource
# # def load_model():
# #     """Loads the HCR-CAF model and weights."""
# #     model = MultiModalCNN(num_classes=NUM_CLASSES, fusion='hcrcaf', seq_length=SEQ_LENGTH, pretr_ef='None')
    
# #     if os.path.exists(MODEL_PATH):
# #         # weights_only=False added to bypass PyTorch 2.6 security blocks for trusted local files
# #         checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=False)
# #         state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
# #         state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
# #         model.load_state_dict(state_dict, strict=False)
# #         model.eval()
# #         return model
# #     else:
# #         st.error(f"Model weights not found at {MODEL_PATH}.")
# #         return None

# # @st.cache_resource
# # def load_face_detector():
# #     """Loads the MTCNN face detector."""
# #     return MTCNN(image_size=224, margin=20, keep_all=False, post_process=False, device='cpu')

# # def process_video(video_path, mtcnn):
# #     """Extracts 15 face frames from the video."""
# #     cap = cv2.VideoCapture(video_path)
# #     frames = []
    
# #     while cap.isOpened():
# #         ret, frame = cap.read()
# #         if not ret: break
# #         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# #         frames.append(Image.fromarray(frame))
# #     cap.release()
    
# #     if not frames: return None
    
# #     indices = np.linspace(0, len(frames) - 1, SEQ_LENGTH, dtype=int)
# #     sampled_frames = [frames[i] for i in indices]
    
# #     face_tensors = []
# #     for img in sampled_frames:
# #         face = mtcnn(img)
# #         if face is not None:
# #             face_tensors.append(face)
# #         else:
# #             face_tensors.append(torch.zeros(3, 224, 224))
# #     # THE FIX: Divide by 255.0 to match the ToTensor() scaling used in training!
# #     # video_tensor = torch.stack(face_tensors, dim=0).permute(1, 0, 2, 3) / 255.0
# #     # return video_tensor.unsqueeze(0)        
# #     # video_tensor = torch.stack(face_tensors, dim=0).permute(1, 0, 2, 3) 
# #     # return video_tensor.unsqueeze(0) 
# #     # 1. Stack and scale to [0, 1]
# #     video_tensor = torch.stack(face_tensors, dim=0) / 255.0 
    
# #     # 2. THE FIX: Apply ImageNet Color Normalization
# #     normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
# #     normalized_frames = []
# #     for i in range(video_tensor.shape[0]):
# #         normalized_frames.append(normalize(video_tensor[i]))
        
# #     # 3. Permute to expected [Channels, Sequence, Height, Width]
# #     video_tensor = torch.stack(normalized_frames, dim=0).permute(1, 0, 2, 3)
    
# #     return video_tensor.unsqueeze(0)

# # def process_audio(audio_path):
# #     """Extracts 10 MFCCs from the dedicated .wav file."""
# #     try:
# #         y, sr = librosa.load(audio_path, sr=22050)
# #         mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10)
# #         audio_tensor = torch.tensor(mfcc, dtype=torch.float32)
# #         return audio_tensor.unsqueeze(0) 
# #     except Exception as e:
# #         st.error(f"Error extracting audio: {e}")
# #         return None

# # # --- UI LAYOUT ---
# # st.title("Multimodal Emotion Recognition 🎭")
# # st.markdown("**Architecture:** Hierarchical Contrastive Residual Cross-Attention Fusion (HCR-CAF)")

# # model = load_model()
# # mtcnn = load_face_detector()

# # st.write("### Upload Modalities")
# # col_vid, col_aud = st.columns(2)

# # with col_vid:
# #     uploaded_video = st.file_uploader("1. Upload Video (.mp4)", type=["mp4"])
# # with col_aud:
# #     uploaded_audio = st.file_uploader("2. Upload Matching Audio (.wav)", type=["wav"])

# # if uploaded_video and uploaded_audio and model:
# #     col1, col2 = st.columns([1, 1])
    
# #     with col1:
# #         st.video(uploaded_video)
# #         st.audio(uploaded_audio)
    
# #     with col2:
# #         st.write("### Processing & Analysis")
# #         analyze_button = st.button("Analyze Multimodal Inputs", type="primary")
        
# #         if analyze_button:
# #             with st.spinner("Extracting features and running HCR-CAF fusion..."):
# #                 # Save uploaded files safely
# #                 with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tv:
# #                     tv.write(uploaded_video.read())
# #                     temp_vid_path = tv.name
                    
# #                 with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as ta:
# #                     ta.write(uploaded_audio.read())
# #                     temp_aud_path = ta.name
                
# #                 # # 1. Preprocess
# #                 # visual_inputs = process_video(temp_vid_path, mtcnn)
# #                 # audio_inputs = process_audio(temp_aud_path)
                
# #                 # if visual_inputs is not None and audio_inputs is not None:
# #                 #     # 2. Inference
# #                 #     with torch.no_grad():
# #                 # 1. Preprocess
# #                 visual_inputs = process_video(temp_vid_path, mtcnn)
# #                 audio_inputs = process_audio(temp_aud_path)
                
# #                 if visual_inputs is not None and audio_inputs is not None:
                    
# #                     # --- THE FIX: Match the training loop shape ---
# #                     # Folds [1, 3, 15, 224, 224] down to [15, 3, 224, 224]
# #                     visual_inputs = visual_inputs.permute(0, 2, 1, 3, 4)
# #                     visual_inputs = visual_inputs.reshape(-1, 3, 224, 224)
# #                     # ----------------------------------------------
                    
# #                     # 2. Inference
# #                     with torch.no_grad():
# #                         logits, _, _ = model(audio_inputs, visual_inputs)
# #                         probabilities = F.softmax(logits, dim=1).squeeze().numpy()
                
                
# #                     # 3. Format Results
# #                     pred_idx = np.argmax(probabilities)
# #                     pred_emotion = EMOTIONS[pred_idx]
# #                     confidence = probabilities[pred_idx] * 100
                    
# #                     st.success(f"**Predicted Emotion:** {pred_emotion} ({confidence:.1f}% Confidence)")
                    
# #                     prob_df = pd.DataFrame({
# #                         'Emotion': EMOTIONS,
# #                         'Confidence (%)': probabilities * 100
# #                     }).sort_values(by='Confidence (%)', ascending=False)
                    
# #                     st.bar_chart(prob_df, x='Emotion', y='Confidence (%)')
                    
# #                 # Safe Windows Cleanup
# #                 try: os.remove(temp_vid_path)
# #                 except PermissionError: pass
                
# #                 try: os.remove(temp_aud_path)
# #                 except PermissionError: pass
# import streamlit as st
# import torch
# import torch.nn.functional as F
# import numpy as np
# import librosa
# import cv2
# import tempfile
# import os
# from facenet_pytorch import MTCNN
# from PIL import Image
# import pandas as pd

# # Import your custom architecture
# from models.multimodalcnn import MultiModalCNN

# # --- CONFIGURATION ---
# MODEL_PATH = "results/RAVDESS_multimodalcnn_15_best0.pth" # Ensure this path is correct!
# NUM_CLASSES = 8
# SEQ_LENGTH = 15
# EMOTIONS = ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised']

# st.set_page_config(page_title="HCR-CAF Emotion Recognition", layout="wide")

# @st.cache_resource
# def load_model():
#     """Loads the HCR-CAF model and weights."""
#     model = MultiModalCNN(num_classes=NUM_CLASSES, fusion='hcrcaf', seq_length=SEQ_LENGTH, pretr_ef='None')
    
#     if os.path.exists(MODEL_PATH):
#         # weights_only=False added to bypass PyTorch 2.6 security blocks for trusted local files
#         checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=False)
#         state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
#         state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
#         model.load_state_dict(state_dict, strict=False)
#         model.eval()
#         return model
#     else:
#         st.error(f"Model weights not found at {MODEL_PATH}.")
#         return None

# @st.cache_resource
# def load_face_detector():
#     """Loads the MTCNN face detector."""
#     return MTCNN(image_size=224, margin=20, keep_all=False, post_process=False, device='cpu')

# def process_video(video_path, mtcnn):
#     """Extracts 15 face frames from the video."""
#     cap = cv2.VideoCapture(video_path)
#     frames = []
    
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret: break
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         frames.append(Image.fromarray(frame))
#     cap.release()
    
#     if not frames: return None
    
#     indices = np.linspace(0, len(frames) - 1, SEQ_LENGTH, dtype=int)
#     sampled_frames = [frames[i] for i in indices]
    
#     face_tensors = []
#     for img in sampled_frames:
#         face = mtcnn(img)
#         if face is not None:
#             face_tensors.append(face)
#         else:
#             # Fallback for blank frames
#             face_tensors.append(torch.zeros(3, 224, 224))
            
#     # THE FIX: Simply stack, permute, and divide by 255.0 to scale to [0, 1]
#     # No ImageNet normalization is used because the model trained from scratch
#     video_tensor = torch.stack(face_tensors, dim=0).permute(1, 0, 2, 3) / 255.0
#     return video_tensor.unsqueeze(0) 

# def process_audio(audio_path):
#     """Extracts 10 MFCCs from the dedicated .wav file."""
#     try:
#         y, sr = librosa.load(audio_path, sr=22050)
#         mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10)
#         audio_tensor = torch.tensor(mfcc, dtype=torch.float32)
#         return audio_tensor.unsqueeze(0) 
#     except Exception as e:
#         st.error(f"Error extracting audio: {e}")
#         return None

# # --- UI LAYOUT ---
# st.title("Multimodal Emotion Recognition 🎭")
# st.markdown("**Architecture:** Hierarchical Contrastive Residual Cross-Attention Fusion (HCR-CAF)")

# model = load_model()
# mtcnn = load_face_detector()

# st.write("### Upload Modalities")
# col_vid, col_aud = st.columns(2)

# with col_vid:
#     uploaded_video = st.file_uploader("1. Upload Original Video (.mp4)", type=["mp4"])
# with col_aud:
#     uploaded_audio = st.file_uploader("2. Upload Matching Audio (.wav)", type=["wav"])

# if uploaded_video and uploaded_audio and model:
#     col1, col2 = st.columns([1, 1])
    
#     with col1:
#         st.video(uploaded_video)
#         st.audio(uploaded_audio)
    
#     with col2:
#         st.write("### Processing & Analysis")
#         analyze_button = st.button("Analyze Multimodal Inputs", type="primary")
        
#         if analyze_button:
#             with st.spinner("Extracting features and running HCR-CAF fusion..."):
                
#                 # Save uploaded files safely
#                 with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tv:
#                     tv.write(uploaded_video.read())
#                     temp_vid_path = tv.name
                    
#                 with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as ta:
#                     ta.write(uploaded_audio.read())
#                     temp_aud_path = ta.name
                
#                 # 1. Preprocess
#                 visual_inputs = process_video(temp_vid_path, mtcnn)
#                 audio_inputs = process_audio(temp_aud_path)
                
#                 if visual_inputs is not None and audio_inputs is not None:
                    
#                     # --- THE DIMENSION FIX ---
#                     # Folds [1, 3, 15, 224, 224] down to [15, 3, 224, 224] for the CNN
#                     visual_inputs = visual_inputs.permute(0, 2, 1, 3, 4)
#                     visual_inputs = visual_inputs.reshape(-1, 3, 224, 224)
#                     # -------------------------
                    
#                     # 2. Inference
#                     with torch.no_grad():
#                         logits, _, _ = model(audio_inputs, visual_inputs)
#                         probabilities = F.softmax(logits, dim=1).squeeze().numpy()
                        
#                     # 3. Format Results
#                     pred_idx = np.argmax(probabilities)
#                     pred_emotion = EMOTIONS[pred_idx]
#                     confidence = probabilities[pred_idx] * 100
                    
#                     st.success(f"**Predicted Emotion:** {pred_emotion} ({confidence:.1f}% Confidence)")
                    
#                     prob_df = pd.DataFrame({
#                         'Emotion': EMOTIONS,
#                         'Confidence (%)': probabilities * 100
#                     }).sort_values(by='Confidence (%)', ascending=False)
                    
#                     st.bar_chart(prob_df, x='Emotion', y='Confidence (%)')
                    
#                 # Safe Windows Cleanup
#                 try: os.remove(temp_vid_path)
#                 except PermissionError: pass
                
#                 try: os.remove(temp_aud_path)
#                 except PermissionError: pass
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
EMOTIONS = ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised']
SEQ_LENGTH = 15

# Custom color palette for the UI
COLOR_MAP = {
    'Happy': '#FFD700', 'Sad': '#1F77B4', 'Angry': '#D62728',
    'Fearful': '#9467BD', 'Disgust': '#2CA02C', 'Surprised': '#FF7F0E',
    'Calm': '#17BECF', 'Neutral': '#7F7F7F'
}

st.set_page_config(page_title="HCR-CAF Emotion Recognition", layout="wide")

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
        st.error(f"Model weights not found at {MODEL_PATH}.")
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
st.title("Multimodal Emotion Recognition 🎭")
st.markdown("**Architecture:** Hierarchical Contrastive Residual Cross-Attention Fusion (HCR-CAF)")

model = load_model()
mtcnn = load_face_detector()

st.write("### Upload Modalities")
col_vid, col_aud = st.columns(2)

with col_vid:
    uploaded_video = st.file_uploader("1. Upload Original Video (.mp4)", type=["mp4"])
with col_aud:
    uploaded_audio = st.file_uploader("2. Upload Matching Audio (.wav)", type=["wav"])

if uploaded_video and uploaded_audio and model:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.video(uploaded_video)
        st.audio(uploaded_audio)
    
    with col2:
        st.write("### Processing & Analysis")
        analyze_button = st.button("Analyze Multimodal Inputs", type="primary", use_container_width=True)
        
        if analyze_button:
            with st.spinner("Extracting features and running HCR-CAF fusion..."):
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tv:
                    tv.write(uploaded_video.read())
                    temp_vid_path = tv.name
                    
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as ta:
                    ta.write(uploaded_audio.read())
                    temp_aud_path = ta.name
                
                visual_inputs = process_video(temp_vid_path, mtcnn)
                audio_inputs = process_audio(temp_aud_path)
                
                if visual_inputs is not None and audio_inputs is not None:
                    visual_inputs = visual_inputs.permute(0, 2, 1, 3, 4)
                    visual_inputs = visual_inputs.reshape(-1, 3, 224, 224)
                    
                    with torch.no_grad():
                        logits, _, _ = model(audio_inputs, visual_inputs)
                        probabilities = F.softmax(logits, dim=1).squeeze().numpy()
                        
                    # Format Results into DataFrame
                    prob_df = pd.DataFrame({
                        'Emotion': EMOTIONS,
                        'Confidence': probabilities * 100
                    }).sort_values(by='Confidence', ascending=False)
                    
                    top_emotion = prob_df.iloc[0]['Emotion']
                    top_conf = prob_df.iloc[0]['Confidence']
                    second_emotion = prob_df.iloc[1]['Emotion']
                    second_conf = prob_df.iloc[1]['Confidence']
                    
                    # --- SMART UI LOGIC ---
                    if top_conf - second_conf < 15.0:
                        # If the top 2 are very close, show a "Blended" warning!
                        st.warning(f"🤔 **Nuanced Expression Detected:** The model sees a blend of **{top_emotion}** ({top_conf:.1f}%) and **{second_emotion}** ({second_conf:.1f}%).")
                    else:
                        st.success(f"🎯 **Primary Prediction:** {top_emotion} ({top_conf:.1f}%)")
                    
                    # --- PLOTLY BEAUTIFUL CHART ---
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
                        xaxis_title="Confidence (%)",
                        yaxis_title="",
                        showlegend=False,
                        height=400,
                        margin=dict(l=0, r=0, t=30, b=0),
                        yaxis={'categoryorder':'total ascending'} # Highest bar at the top
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                try: os.remove(temp_vid_path)
                except: pass
                try: os.remove(temp_aud_path)
                except: pass