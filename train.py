import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
from models.multimodalcnn import MultiModalCNN
import json
from tqdm import tqdm

# --- CONFIGURATION (Based on opts1774280244.99898890.json) ---
CONFIG = {
    "annotation_path": "data/annotations.txt",
    "base_dir": "./",
    "model_path": "results/RAVDESS_multimodalcnn_15_best0.pth",
    "num_classes": 8,
    "seq_length": 15,
    "batch_size": 16,
    "learning_rate": 1e-4, # Lower for fine-tuning
    "epochs": 10,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# --- DATASET LOADER ---
class EmotionDataset(Dataset):
    def __init__(self, annotation_path, base_dir, subset='training'):
        self.base_dir = base_dir
        self.samples = []
        
        # Parse annotations (Format: video_path;audio_path;label;subset)
        if not os.path.exists(annotation_path):
            print(f"Warning: Annotation file {annotation_path} not found. Ensure you run preprocessing first.")
            return
            
        with open(annotation_path, 'r') as f:
            for line in f:
                vid_path, aud_path, label, sub = line.strip().split(';')
                if sub == subset:
                    self.samples.append((vid_path, aud_path, int(label)))
                    
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        vid_path, aud_path, label = self.samples[idx]
        
        full_vid_path = os.path.join(self.base_dir, vid_path)
        full_aud_path = os.path.join(self.base_dir, aud_path)
        
        # 1. Load Visual Modality (.npy)
        # Shape: [15, 224, 224, 3] -> Permute to [15, 3, 224, 224] for CNN
        try:
            video_frames = np.load(full_vid_path)
            video_tensor = torch.tensor(video_frames, dtype=torch.float32) / 255.0
            visual_input = video_tensor.permute(0, 3, 1, 2)
        except Exception as e:
            # Fallback for missing/corrupted npy
            visual_input = torch.zeros((15, 3, 224, 224), dtype=torch.float32)
            
        # 2. Load Audio Modality (.wav)
        # Extract 10 MFCCs as done in preprocessing/app.py
        try:
            y, sr = librosa.load(full_aud_path, sr=22050)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10)
            audio_input = torch.tensor(mfcc, dtype=torch.float32)
        except Exception as e:
            # Fallback for missing/corrupted wav
            audio_input = torch.zeros((10, 156), dtype=torch.float32) # Approx size for 3.6s
            
        # 3. Dummy Lexical Stream
        # RAVDESS has no text, so we pass a static dummy string. RoBERTa will encode this 
        # into a neutral vector, and the cross-attention will learn to ignore it.
        lexical_input = "[NO_SPEECH]"
            
        return lexical_input, audio_input, visual_input, torch.tensor(label, dtype=torch.long)


# --- FINE TUNING ENGINE ---
def train_model():
    print(f"--- Starting Fine-Tuning on {CONFIG['device']} ---")
    
    # 1. Load Datasets
    train_dataset = EmotionDataset(CONFIG["annotation_path"], CONFIG["base_dir"], subset='training')
    val_dataset = EmotionDataset(CONFIG["annotation_path"], CONFIG["base_dir"], subset='validation')
    
    if len(train_dataset) == 0:
        print("Error: No training data found. Have you extracted the RAVDESS dataset into data/ ?")
        return
        
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2)
    
    print(f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")
    
    # 2. Load Pre-trained Model
    model = MultiModalCNN(num_classes=CONFIG["num_classes"], fusion='hcrcaf', seq_length=CONFIG["seq_length"], pretr_ef='None')
    if os.path.exists(CONFIG["model_path"]):
        checkpoint = torch.load(CONFIG["model_path"], map_location=CONFIG["device"], weights_only=False)
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        print(f"Successfully loaded base weights from {CONFIG['model_path']}")
    else:
        print("Warning: Base weights not found! Training from scratch.")
        
    model = model.to(CONFIG["device"])
    
    # 3. Freeze Early Layers (Optional but recommended for Fine-tuning)
    for param in model.audio_model.parameters():
        param.requires_grad = False
    for param in model.visual_model.parameters():
        param.requires_grad = False
        
    # We only train the HCR-CAF fusion layers and the final classifier!
    print("Frozen early acoustic and visual feature extractors. Fine-tuning Fusion and Classifier heads only.")
    
    # 4. Setup Optimizer & Loss
    # We only pass parameters that require gradients to the optimizer
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=CONFIG["learning_rate"], weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # 5. Training Loop
    best_val_loss = float('inf')
    
    for epoch in range(CONFIG["epochs"]):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        print(f"\nEpoch {epoch+1}/{CONFIG['epochs']}")
        train_bar = tqdm(train_loader, desc="Training")
        
        for lexical_inputs, audio_inputs, visual_inputs, labels in train_bar:
            # Note: lexical_inputs is a list of strings, so it doesn't need .to(device)
            audio_inputs = audio_inputs.to(CONFIG["device"])
            visual_inputs = visual_inputs.to(CONFIG["device"])
            labels = labels.to(CONFIG["device"])
            
            optimizer.zero_grad()
            
            # Forward pass (Tri-Modal)
            logits, _, _, _, _ = model(lexical_inputs, visual_inputs, audio_inputs)
            loss = criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Metrics
            running_loss += loss.item() * audio_inputs.size(0)
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            train_bar.set_postfix({'loss': loss.item(), 'acc': 100 * correct / total})
            
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = 100 * correct / total
        print(f"Train Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%")
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for lexical_inputs, audio_inputs, visual_inputs, labels in tqdm(val_loader, desc="Validating"):
                audio_inputs = audio_inputs.to(CONFIG["device"])
                visual_inputs = visual_inputs.to(CONFIG["device"])
                labels = labels.to(CONFIG["device"])
                
                logits, _, _, _, _ = model(lexical_inputs, visual_inputs, audio_inputs)
                loss = criterion(logits, labels)
                
                val_loss += loss.item() * audio_inputs.size(0)
                _, predicted = torch.max(logits, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        val_epoch_loss = val_loss / len(val_dataset)
        val_epoch_acc = 100 * val_correct / val_total
        print(f"Val Loss: {val_epoch_loss:.4f} | Val Acc: {val_epoch_acc:.2f}%")
        
        # Save Best Model
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            save_path = f"results/finetuned_model_epoch{epoch+1}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"🏆 New best model saved to {save_path}")

if __name__ == "__main__":
    train_model()
