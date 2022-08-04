import torch.nn as nn 

from .block import MLP
from utils.nested_tensor import NestedTensor

class SPTS(nn.Module):
    def __init__(self, backbone, transformer, num_classes):
        super(SPTS, self).__init__()
        self.backbone = backbone 
        self.transformer = transformer     
        self.input_proj = nn.Conv2d(backbone.num_channels, transformer.d_model, kernel_size=1)
        self.vocab_embed = MLP(transformer.d_model, transformer.d_model, num_classes, 3)

    def forward(self, samples: NestedTensor, sequence):
        features, pos = self.backbone(samples)
        src, mask = features[-1].decompose()
        assert mask is not None
        
        out = self.transformer(self.input_proj(src), mask, pos[-1], sequence, self.vocab_embed)
        return out
