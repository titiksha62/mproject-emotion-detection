import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
from models.core.multimodal_tri import MultiModalCNNTri
from src.pipeline.partial_loader import load_partial_weights
import json
from tqdm import tqdm

# --- CONFIGURATION ---
CONFIG = {
    "annotation_path": "data/annotations.txt",
    "base_dir": "./",
    "model_path": "results/RAVDESS_multimodalcnn_15_best0.pth",
    "num_classes": 8,
    "embed_dim": 128,
    "batch_size": 16,
    "learning_rate": 5e-5, # Extremely low for fine-tuning
    "weight_decay": 1e-2, # Heavy L2 Regularization (Anti-Overfitting)
    "epochs": 20,
    "patience": 5, # Early stopping patience
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# --- DATASET LOADER ---
class EmotionDataset(Dataset):
    def __init__(self, annotation_path, base_dir, subset='training'):
        self.base_dir = base_dir
        self.samples = []
        
        if not os.path.exists(annotation_path):
            print(f"Warning: Annotation file {annotation_path} not found.")
            return
            
        with open(annotation_path, 'r') as f:
            for line in f:
                vid_path, aud_path, label, sub = line.strip().split(';')
                if sub == subset:
                    self.samples.append((vid_path, aud_path, int(label)))
                    
    def __len__(self): return len(self.samples)
        
    def __getitem__(self, idx):
        vid_path, aud_path, label = self.samples[idx]
        full_vid_path = os.path.join(self.base_dir, vid_path)
        full_aud_path = os.path.join(self.base_dir, aud_path)
        
        # 1. Load Visual
        try:
            video_frames = np.load(full_vid_path)
            video_tensor = torch.tensor(video_frames, dtype=torch.float32) / 255.0
            visual_input = video_tensor.permute(0, 3, 1, 2)
        except:
            visual_input = torch.zeros((15, 3, 224, 224), dtype=torch.float32)
            
        # 2. Load Acoustic
        try:
            y, sr = librosa.load(full_aud_path, sr=22050)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10)
            audio_input = torch.tensor(mfcc, dtype=torch.float32)
        except:
            audio_input = torch.zeros((10, 156), dtype=torch.float32)
            
        # 3. Dummy Lexical (No text in RAVDESS)
        lexical_input = "[NO_SPEECH]"
            
        return lexical_input, audio_input, visual_input, torch.tensor(label, dtype=torch.long)

# --- FINE TUNING ENGINE ---
def train_model():
    print(f"--- Starting Tri-Modal Fine-Tuning on {CONFIG['device']} ---")
    
    # 1. Load Data
    train_dataset = EmotionDataset(CONFIG["annotation_path"], CONFIG["base_dir"], subset='training')
    val_dataset = EmotionDataset(CONFIG["annotation_path"], CONFIG["base_dir"], subset='validation')
    
    if len(train_dataset) == 0:
        print("Error: No training data found.")
        return
        
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2)
    
    # 2. Load Model & Partial Weights
    model = MultiModalCNNTri(num_classes=CONFIG["num_classes"], embed_dim=CONFIG["embed_dim"], device=CONFIG["device"])
    
    if os.path.exists(CONFIG["model_path"]):
        print("Injecting Base Weights...")
        load_partial_weights(model, CONFIG["model_path"], device=CONFIG["device"])
    else:
        print("Warning: Base weights not found! Training entirely from scratch.")
        
    model = model.to(CONFIG["device"])
    
    # 3. Anti-Overfitting: Aggressive Freezing
    # Freeze Unimodal Extractors. Only train Fusion and Classifier.
    for param in model.lexical_model.parameters(): param.requires_grad = False
    for param in model.visual_model.parameters(): param.requires_grad = False
    for param in model.acoustic_model.parameters(): param.requires_grad = False
    print("[SAFEGUARD] Unimodal Extractors frozen. Fine-tuning Fusion and Classifier heads only.")
    
    # 4. Setup Optimizer & Loss
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                            lr=CONFIG["learning_rate"], 
                            weight_decay=CONFIG["weight_decay"])
    criterion = nn.CrossEntropyLoss()
    
    # 5. Training Loop with Early Stopping
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(CONFIG["epochs"]):
        model.train()
        running_loss = 0.0
        correct = 0; total = 0
        
        print(f"\nEpoch {epoch+1}/{CONFIG['epochs']}")
        train_bar = tqdm(train_loader, desc="Training")
        
        for lexical_inputs, audio_inputs, visual_inputs, labels in train_bar:
            audio_inputs = audio_inputs.to(CONFIG["device"])
            visual_inputs = visual_inputs.to(CONFIG["device"])
            labels = labels.to(CONFIG["device"])
            
            optimizer.zero_grad()
            logits, _, _, _, _ = model(lexical_inputs, visual_inputs, audio_inputs)
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * audio_inputs.size(0)
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            train_bar.set_postfix({'loss': loss.item(), 'acc': 100 * correct / total})
            
        print(f"Train Loss: {running_loss / len(train_dataset):.4f} | Acc: {100 * correct / total:.2f}%")
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_correct = 0; val_total = 0
        
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
        
        # Early Stopping Logic
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            epochs_no_improve = 0
            save_path = f"results/finetuned_trimodal_best.pth"
            torch.save(model.state_dict(), save_path)
            print(f"🏆 Best model saved to {save_path}")
        else:
            epochs_no_improve += 1
            print(f"⚠️ No improvement for {epochs_no_improve} epochs.")
            if epochs_no_improve >= CONFIG["patience"]:
                print("🛑 Early Stopping triggered to prevent overfitting!")
                break

if __name__ == "__main__":
    train_model()
