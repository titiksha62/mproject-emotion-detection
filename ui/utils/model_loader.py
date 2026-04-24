import torch
import streamlit as st
import os
from facenet_pytorch import MTCNN
from models.multimodalcnn import MultiModalCNN

MODEL_PATH = "results/RAVDESS_multimodalcnn_15_best0.pth"
NUM_CLASSES = 8
SEQ_LENGTH = 15

@st.cache_resource
def load_model():
    model = MultiModalCNN(num_classes=NUM_CLASSES, fusion='hcrcaf', seq_length=SEQ_LENGTH, pretr_ef='None')

    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model
    else:
        st.error("Model file not found!")
        return None

@st.cache_resource
def load_face_detector():
    return MTCNN(image_size=224, margin=20, keep_all=False, post_process=False, device='cpu')