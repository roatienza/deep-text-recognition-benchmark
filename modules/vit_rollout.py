"""
Reference:
https://jacobgil.github.io/deeplearning/vision-transformer-explainability
"""

import torch
import numpy
import sys
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image

def rollout(attentions, discard_ratio, head_fusion, token=0, seqlen=25):
    result = torch.eye(attentions[0].size(-1))
    with torch.no_grad():
        for attention in attentions:
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=1)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(axis=1)[0]
            elif head_fusion == "min":
                attention_heads_fused = attention.min(axis=1)[0]
            else:
                raise "Attention head fusion type Not supported"

            # Drop the lowest attentions, but
            # don't drop the class token
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False)
            #indices = indices[indices != 0]
            flat[0, indices] = 0

            I = torch.eye(attention_heads_fused.size(-1))
            a = (attention_heads_fused + 1.0*I)/2
            #a = a / a.sum(dim=-1)

            result = torch.matmul(a, result)
    
    #result = result / result.sum(dim=-1)
    # Look at the total attention between the class token,
    # and the image patches
    # mask = result[0, 1:, 1:]
    if token<=0:
        mask = result[0, :seqlen, 1:]
        mask = torch.mean(mask, dim = 0)
    else:
        mask = result[0, :seqlen, :]
        mask = mask[token, 1:]
    # In case of 224x224 image, this brings us from 196 to 14
    width = int(mask.size(-1)**0.5)
    mask = mask.reshape(width, width).numpy()
    mask = mask / np.max(mask)
    return mask    

class VITAttentionRollout:
    def __init__(self,
                 model, 
                 attention_layer_name='attn_drop', 
                 head_fusion="mean",
                 discard_ratio=0.9):
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        for name, module in self.model.named_modules():
            if attention_layer_name in name:
                module.register_forward_hook(self.get_attention)

        self.attentions = []

    def get_attention(self, module, input, output):
        self.attentions.append(output)

    def __call__(self, input_tensor, token, seqlen):
        self.attentions = []
        with torch.no_grad():
            output = self.model(input_tensor, "")

        return rollout(self.attentions,
                       self.discard_ratio, 
                       self.head_fusion, 
                       token, 
                       seqlen)
