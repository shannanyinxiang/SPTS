import torch.nn as nn 

from typing import List
from utils.nested_tensor import NestedTensor

class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super(Joiner, self).__init__(backbone, position_embedding)
    
    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for _, x in xs.items():
            out.append(x)
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos
