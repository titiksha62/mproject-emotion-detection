import torch

def load_partial_weights(model, pth_path, device='cpu'):
    """
    Safely loads weights from a legacy (Bi-Modal) .pth file into the new 
    Tri-Modal architecture. It extracts only the overlapping keys (Visual & Acoustic)
    and ignores missing/unexpected keys to prevent shape mismatch crashes.
    """
    try:
        checkpoint = torch.load(pth_path, map_location=device, weights_only=False)
        # Handle dict wrapping if necessary
        state_dict = checkpoint.get('state_dict', checkpoint)
        
        # Clean module prefixes if trained with DataParallel
        clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        # Filter out incompatible keys (like the old classifier and fusion block)
        model_dict = model.state_dict()
        pretrained_dict = {}
        skipped_keys = []
        loaded_keys = []
        
        for k, v in clean_state_dict.items():
            if k in model_dict and v.shape == model_dict[k].shape:
                pretrained_dict[k] = v
                loaded_keys.append(k)
            else:
                skipped_keys.append(k)
                
        # Update current model dict
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        
        print(f"[PARTIAL LOADER] Successfully loaded {len(loaded_keys)} matching layers.")
        print(f"[PARTIAL LOADER] Skipped {len(skipped_keys)} incompatible layers (Fusion/Classifier).")
        return True
        
    except Exception as e:
        print(f"[PARTIAL LOADER] Error loading weights: {e}")
        return False
