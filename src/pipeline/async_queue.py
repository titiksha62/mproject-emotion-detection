import asyncio
from concurrent.futures import ThreadPoolExecutor
import torch

class AsyncStreamQueue:
    """
    Simulates AWS SQS / EventBridge asynchronous event routing.
    Since we are running on a CPU-only environment, executing 3 heavy deep learning
    models sequentially would cause UI blocking and massive delivery time.
    We use ThreadPoolExecutor and asyncio to run them in parallel on separate threads.
    """
    def __init__(self, max_workers=3):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
    async def process_visual(self, visual_model, x_frames):
        loop = asyncio.get_running_loop()
        # Offload heavy CPU tensor math to a background thread
        return await loop.run_in_executor(self.executor, visual_model, x_frames)
        
    async def process_acoustic(self, acoustic_model, x_audio):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, acoustic_model, x_audio)
        
    async def process_lexical(self, lexical_model, text_list):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, lexical_model, text_list)
        
    async def dispatch_parallel(self, model_core, text_list, x_frames, x_audio):
        """
        Dispatches all 3 unimodal extractions concurrently, waits for them, 
        and then passes them to the fusion engine.
        """
        print("[QUEUE] Dispatching Unimodal Extraction to parallel CPU threads...")
        
        # Fire all three tasks concurrently
        task_v = self.process_visual(model_core.visual_model, x_frames)
        task_a = self.process_acoustic(model_core.acoustic_model, x_audio)
        task_l = self.process_lexical(model_core.lexical_model, text_list)
        
        # Wait for all to complete
        v_seq, a_seq, l_seq = await asyncio.gather(task_v, task_a, task_l)
        
        print("[QUEUE] Streams synchronized. Routing to HCR-CAF Fusion Engine...")
        
        # Now pass pre-extracted sequences to fusion (Bypassing the standard forward method)
        l_seq = l_seq.mean(dim=1, keepdim=True)
        v_seq = v_seq.mean(dim=1, keepdim=True)
        a_seq = a_seq.mean(dim=1, keepdim=True)
        
        l_fused, v_fused, a_fused = model_core.fusion_block(l_seq, v_seq, a_seq)
        
        l_pooled = l_fused.squeeze(1)
        v_pooled = v_fused.squeeze(1)
        a_pooled = a_fused.squeeze(1)
        
        joint_representation = torch.cat([l_pooled, v_pooled, a_pooled], dim=-1)
        logits = model_core.classifier(joint_representation)
        
        return logits
