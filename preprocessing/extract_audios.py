# preprocessing/extract_audios.py
import librosa
import os
import soundfile as sf
import numpy as np
from tqdm import tqdm

root = '../data/Audio_Speech_Actors/'
target_time = 3.6 #sec

print("Processing audio files...")
for actor in tqdm(os.listdir(root)):
    actor_path = os.path.join(root, actor)
    if not os.path.isdir(actor_path): continue
    
    for audiofile in os.listdir(actor_path):
        if not audiofile.endswith('.wav') or 'croppad' in audiofile:
            continue

        audios = librosa.core.load(os.path.join(actor_path, audiofile), sr=22050)
        y = audios[0]
        sr = audios[1]
        target_length = int(sr * target_time)
        
        if len(y) < target_length:
            y = np.array(list(y) + [0 for i in range(target_length - len(y))])
        else:
            remain = len(y) - target_length
            y = y[remain//2:-(remain - remain//2)]
        
        sf.write(os.path.join(actor_path, audiofile[:-4]+'_croppad.wav'), y, sr)
print("Audio processing complete!")