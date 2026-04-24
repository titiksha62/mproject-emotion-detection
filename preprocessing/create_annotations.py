# preprocessing/create_annotations.py
import os

root_video = '../data/Video_Speech_Actors/'
root_audio = '../data/Audio_Speech_Actors/'
output_file = '../data/annotations.txt'

# Original fold split from the repo (Train/Val/Test by Actor ID)
folds = [[[0,1,2,3],[4,5,6,7],[8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]]]
test_ids, val_ids, train_ids = folds[0]

print("Generating annotations.txt...")
with open(output_file, 'w') as f:
    for i, actor in enumerate(sorted(os.listdir(root_video))):
        actor_vid_path = os.path.join(root_video, actor)
        if not os.path.isdir(actor_vid_path): continue

        for video in os.listdir(actor_vid_path):
            if not video.endswith('.npy') or 'croppad' not in video:
                continue
            
            label = str(int(video.split('-')[2]))
            
            # RAVDESS audio files start with '03' instead of '01' (video) or '02' (audio-only)
            audio = '03' + video.split('_face')[0][2:] + '_croppad.wav'  
            
            # Write relative paths that will survive being zipped and moved to Kaggle
            rel_vid_path = f"data/video/{actor}/{video}"
            rel_aud_path = f"data/audio/{actor}/{audio}"
            
            if i in train_ids:
                subset = 'training'
            elif i in val_ids:
                subset = 'validation'
            else:
                subset = 'testing'
                
            f.write(f"{rel_vid_path};{rel_aud_path};{label};{subset}\n")

print(f"Annotations saved to {output_file}!")