import torch
from .spts import SPTS
from .backbone import build_backbone
from .transformer import build_transformer

def build_model(args):
    backbone = build_backbone(args)
    transformer = build_transformer(args)
    model = SPTS(backbone, transformer, args.num_classes)

    device = torch.device('cuda')
    model = model.to(device)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
    
    return model