# preprocessing/extract_faces.py
import os
import numpy as np          
import cv2
from tqdm import tqdm
import torch
from facenet_pytorch import MTCNN

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(image_size=(720, 1280), device=device)

save_frames = 15
input_fps = 30
save_length = 3.6 #seconds
save_avi = False # Set to False to save massive amounts of CPU time
# save_avi = True 

failed_videos = []
root = '../data/Video_Speech_Actors/'

select_distributed = lambda m, n: [i*n//m + n//(2*m) for i in range(m)]

print(f"Processing video faces on device: {device}...")
for sess in tqdm(sorted(os.listdir(root))):   
    sess_path = os.path.join(root, sess)
    if not os.path.isdir(sess_path): continue
    
    for filename in os.listdir(sess_path):
        if filename.endswith('.mp4'):
    # for filename in os.listdir(sess_path):
    #     if filename.endswith('.mp4'):
            # --- ADD THESE 3 LINES RIGHT HERE ---
            npy_target = os.path.join(sess_path, filename[:-4]+'_facecroppad.npy')
            if os.path.exists(npy_target):
                continue # Skip this video, we already did it!
            # ------------------------------------
            # cap = cv2.VideoCapture(os.path.join(sess_path, filename))
            cap = cv2.VideoCapture(os.path.join(sess_path, filename))  
            framen = 0
            while True:
                i,q = cap.read()
                if not i: break
                framen += 1
            
            cap = cv2.VideoCapture(os.path.join(sess_path, filename))

            if save_length*input_fps > framen:                    
                skip_begin = int((framen - (save_length*input_fps)) // 2)
                for i in range(skip_begin):
                    _, im = cap.read() 
                    
            framen = int(save_length*input_fps)    
            frames_to_select = select_distributed(save_frames,framen)

            numpy_video = []
            frame_ctr = 0
            
            while True: 
                ret, im = cap.read()
                if not ret: break
                if frame_ctr not in frames_to_select:
                    frame_ctr += 1
                    continue
                else:
                    frames_to_select.remove(frame_ctr)
                    frame_ctr += 1

                try:
                    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                except:
                    failed_videos.append((sess, filename))
                    break
        
                temp = im[:,:,-1]
                im_rgb = im.copy()
                im_rgb[:,:,-1] = im_rgb[:,:,0]
                im_rgb[:,:,0] = temp
                im_rgb = torch.tensor(im_rgb).to(device)

                bbox = mtcnn.detect(im_rgb)
                if bbox[0] is not None:
                    bbox = bbox[0][0]
                    bbox = [round(x) for x in bbox]
                    x1, y1, x2, y2 = bbox
                    # Simple boundary check
                    y1, y2 = max(0, y1), max(0, y2)
                    x1, x2 = max(0, x1), max(0, x2)
                    im = im[y1:y2, x1:x2, :]
                
                # If face detection fails, it resizes the whole frame instead of crashing
                im = cv2.resize(im, (224,224)) 
                numpy_video.append(im)
                
            if len(frames_to_select) > 0:
                for i in range(len(frames_to_select)):
                    numpy_video.append(np.zeros((224,224,3), dtype=np.uint8))
                    
            np.save(os.path.join(sess_path, filename[:-4]+'_facecroppad.npy'), np.array(numpy_video))
            if len(numpy_video) != 15:
                print('Error: Length mismatch in', sess, filename)    

print("Failed videos:", failed_videos)
print("Video processing complete!")