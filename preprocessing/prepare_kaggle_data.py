# preprocessing/prepare_kaggle_data.py
import os
import shutil
from tqdm import tqdm

src_dir = '../data'
dest_dir = '../kaggle_dataset'

print("Creating lightweight Kaggle dataset folder...")
os.makedirs(dest_dir, exist_ok=True)

# Your exact folder names
audio_dir_name = 'Audio_Speech_Actors'
video_dir_name = 'Video_Speech_Actors'

os.makedirs(os.path.join(dest_dir, audio_dir_name), exist_ok=True)
os.makedirs(os.path.join(dest_dir, video_dir_name), exist_ok=True)

# 1. Copy annotations
annotations_src = os.path.join(src_dir, 'annotations.txt')
if os.path.exists(annotations_src):
    shutil.copy(annotations_src, os.path.join(dest_dir, 'annotations.txt'))
    print("Copied annotations.txt")
else:
    print("Warning: annotations.txt not found in data folder!")

# 2. Copy ONLY the processed .npy and _croppad.wav files
for modality_dir in [audio_dir_name, video_dir_name]:
    src_modality_path = os.path.join(src_dir, modality_dir)
    dest_modality_path = os.path.join(dest_dir, modality_dir)
    
    if not os.path.exists(src_modality_path):
        print(f"Skipping {modality_dir} - folder not found.")
        continue
        
    for actor in tqdm(os.listdir(src_modality_path), desc=f"Copying {modality_dir}"):
        actor_src = os.path.join(src_modality_path, actor)
        actor_dest = os.path.join(dest_modality_path, actor)
        
        if not os.path.isdir(actor_src): 
            continue
            
        os.makedirs(actor_dest, exist_ok=True)
        
        for file in os.listdir(actor_src):
            # ONLY grab the files the AI actually uses
            if file.endswith('.npy') or file.endswith('_croppad.wav'):
                shutil.copy(os.path.join(actor_src, file), os.path.join(actor_dest, file))

print(f"\nDone! Your clean Kaggle dataset is ready in: {dest_dir}")