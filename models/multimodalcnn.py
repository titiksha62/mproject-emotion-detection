# %%writefile models/multimodalcnn.py
import torch
import torch.nn as nn
from models.modulator import Modulator
from models.efficientface import LocalFeatureExtractor, InvertedResidual
from models.transformer_timm import AttentionBlock, Attention

def conv1d_block(in_channels, out_channels, kernel_size=3, stride=1, padding='same'):
    return nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,stride=stride, padding=padding),nn.BatchNorm1d(out_channels),nn.ReLU(inplace=True)) 

class EfficientFaceTemporal(nn.Module):
    def __init__(self, stages_repeats, stages_out_channels, num_classes=8, im_per_sample=15):
        super(EfficientFaceTemporal, self).__init__()
        self._stage_out_channels = stages_out_channels
        self.conv1 = nn.Sequential(nn.Conv2d(3, self._stage_out_channels[0], 3, 2, 1, bias=False),nn.BatchNorm2d(self._stage_out_channels[0]),nn.ReLU(inplace=True),)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
        input_channels = self._stage_out_channels[0]
        for name, repeats, output_channels in zip(stage_names, stages_repeats, self._stage_out_channels[1:]):
            seq = [InvertedResidual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(InvertedResidual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels
        self.local = LocalFeatureExtractor(29, 116, 1)
        self.modulator = Modulator(116)
        output_channels = self._stage_out_channels[-1]
        self.conv5 = nn.Sequential(nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),nn.BatchNorm2d(output_channels),nn.ReLU(inplace=True),)
        self.conv1d_0 = conv1d_block(output_channels, 64)
        self.conv1d_1 = conv1d_block(64, 64)
        self.conv1d_2 = conv1d_block(64, 128)
        self.conv1d_3 = conv1d_block(128, 128)
        self.classifier_1 = nn.Sequential(nn.Linear(128, num_classes))
        self.im_per_sample = im_per_sample
        
    def forward_features(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.modulator(self.stage2(x)) + self.local(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = x.mean([2, 3]) 
        return x

    def forward_stage1(self, x):
        n_samples = x.shape[0] // self.im_per_sample
        x = x.view(n_samples, self.im_per_sample, x.shape[1])
        x = x.permute(0,2,1)
        x = self.conv1d_0(x)
        x = self.conv1d_1(x)
        return x
        
    def forward_stage2(self, x):
        x = self.conv1d_2(x)
        x = self.conv1d_3(x)
        return x

def init_feature_extractor(model, path):
    if path == 'None' or path is None: return
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    pre_trained_dict = checkpoint['state_dict']
    pre_trained_dict = {key.replace("module.", ""): value for key, value in pre_trained_dict.items()}
    model.load_state_dict(pre_trained_dict, strict=False)

def conv1d_block_audio(in_channels, out_channels, kernel_size=3, stride=1, padding='same'):
    return nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,stride=stride, padding='valid'),nn.BatchNorm1d(out_channels),nn.ReLU(inplace=True), nn.MaxPool1d(2,1))

class AudioCNNPool(nn.Module):
    def __init__(self, num_classes=8):
        super(AudioCNNPool, self).__init__()
        self.conv1d_0 = conv1d_block_audio(10, 64)
        self.conv1d_1 = conv1d_block_audio(64, 128)
        self.conv1d_2 = conv1d_block_audio(128, 256)
        self.conv1d_3 = conv1d_block_audio(256, 128)
            
    def forward_stage1(self,x):            
        x = self.conv1d_0(x)
        x = self.conv1d_1(x)
        return x
    
    def forward_stage2(self,x):
        x = self.conv1d_2(x)
        x = self.conv1d_3(x)   
        return x

class MultiModalCNN(nn.Module):
    def __init__(self, num_classes=8, fusion='ia', seq_length=15, pretr_ef='None', num_heads=1):
        super(MultiModalCNN, self).__init__()
        self.audio_model = AudioCNNPool(num_classes=num_classes)
        self.visual_model = EfficientFaceTemporal([4, 8, 4], [29, 116, 232, 464, 1024], num_classes, seq_length)
        init_feature_extractor(self.visual_model, pretr_ef)
                           
        e_dim = 128
        self.fusion = fusion

        if fusion == 'lt':
            self.av = AttentionBlock(in_dim_k=128, in_dim_q=128, out_dim=e_dim, num_heads=num_heads)
            self.va = AttentionBlock(in_dim_k=128, in_dim_q=128, out_dim=e_dim, num_heads=num_heads)
        elif fusion == 'ia':
            self.av1 = Attention(in_dim_k=64, in_dim_q=128, out_dim=128, num_heads=num_heads)
            self.va1 = Attention(in_dim_k=128, in_dim_q=64, out_dim=64, num_heads=num_heads)
        
        # --- HCR-CAF INIT ---
        elif fusion == 'hcrcaf':
            from models.hcrcaf import HCRCAFFusion
            self.hcrcaf_block = HCRCAFFusion(embed_dim=e_dim, num_heads=4)
            
        self.classifier_1 = nn.Sequential(nn.Linear(e_dim*2, num_classes))

    def forward(self, x_audio, x_visual):
        if self.fusion == 'lt': return self.forward_transformer(x_audio, x_visual)
        elif self.fusion == 'ia': return self.forward_feature_2(x_audio, x_visual)
        elif self.fusion == 'hcrcaf': return self.forward_hcrcaf(x_audio, x_visual)

    # --- UPDATED HCR-CAF ROUTING (Fixed Dimensions) ---
    def forward_hcrcaf(self, x_audio, x_visual):
        # 1. Pass through stage 1
        x_audio = self.audio_model.forward_stage1(x_audio)
        x_visual = self.visual_model.forward_features(x_visual)
        x_visual = self.visual_model.forward_stage1(x_visual)

        # 2. Pass through stage 2 FIRST (This forces both to exactly 128 dimensions)
        x_audio = self.audio_model.forward_stage2(x_audio)       
        x_visual = self.visual_model.forward_stage2(x_visual)

        # 3. Permute for Attention (Batch, Seq, Feature)
        proj_x_a = x_audio.permute(0,2,1)
        proj_x_v = x_visual.permute(0,2,1)

        # 4. Apply your HCR-CAF
        fused_a, fused_v = self.hcrcaf_block(proj_x_a, proj_x_v)
        
        # 5. Temporal Pooling
        audio_pooled = fused_a.mean(dim=1) 
        video_pooled = fused_v.mean(dim=1)

        # 6. Classification
        x = torch.cat((audio_pooled, video_pooled), dim=-1)
        logits = self.classifier_1(x)
        
        return logits, video_pooled, audio_pooled

    # (Baseline fusions kept intact)
    def forward_feature_2(self, x_audio, x_visual):
        x_audio = self.audio_model.forward_stage1(x_audio)
        x_visual = self.visual_model.forward_features(x_visual)
        x_visual = self.visual_model.forward_stage1(x_visual)
        _, h_av = self.av1(x_visual.permute(0,2,1), x_audio.permute(0,2,1))
        _, h_va = self.va1(x_audio.permute(0,2,1), x_visual.permute(0,2,1))
        if h_av.size(1) > 1: h_av = torch.mean(h_av, axis=1).unsqueeze(1)
        if h_va.size(1) > 1: h_va = torch.mean(h_va, axis=1).unsqueeze(1)
        x_audio = h_va.sum([-2])*x_audio
        x_visual = h_av.sum([-2])*x_visual
        x_audio = self.audio_model.forward_stage2(x_audio)       
        x_visual = self.visual_model.forward_stage2(x_visual)
        x = torch.cat((x_audio.mean([-1]), x_visual.mean([-1])), dim=-1)
        return self.classifier_1(x)

    def forward_transformer(self, x_audio, x_visual):
        x_audio = self.audio_model.forward_stage1(x_audio)
        proj_x_a = self.audio_model.forward_stage2(x_audio)
        x_visual = self.visual_model.forward_features(x_visual) 
        x_visual = self.visual_model.forward_stage1(x_visual)
        proj_x_v = self.visual_model.forward_stage2(x_visual)
        h_av = self.av(proj_x_v.permute(0, 2, 1), proj_x_a.permute(0, 2, 1))
        h_va = self.va(proj_x_a.permute(0, 2, 1), proj_x_v.permute(0, 2, 1))
        x = torch.cat((h_av.mean([1]), h_va.mean([1])), dim=-1)  
        return self.classifier_1(x)