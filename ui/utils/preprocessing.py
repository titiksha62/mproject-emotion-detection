import torch
import numpy as np
import librosa
import cv2
from PIL import Image

SEQ_LENGTH = 15

def process_video(video_path, mtcnn):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))
    cap.release()

    if not frames:
        return None

    indices = np.linspace(0, len(frames)-1, SEQ_LENGTH, dtype=int)
    sampled = [frames[i] for i in indices]

    faces = []
    for img in sampled:
        face = mtcnn(img)
        if face is not None:
            faces.append(face)
        else:
            faces.append(torch.zeros(3,224,224))

    video_tensor = torch.stack(faces).permute(1,0,2,3) / 255.0
    return video_tensor.unsqueeze(0)


def process_audio(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=22050)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10)
        return torch.tensor(mfcc).float().unsqueeze(0)
    except:
        return None