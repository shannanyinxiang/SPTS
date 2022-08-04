import torch
import torch.nn as nn

from .lr_scheduler import LinearDecayLR

def build_criterion(args):
    weight = torch.ones(args.num_classes)
    weight[args.eos_index] = args.eos_loss_coef
    criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=args.padding_index)

    device = torch.device('cuda')
    criterion = criterion.to(device)
    return criterion

def build_optimizer(model, args):
    if args.distributed:
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    backbone_lr = args.lr * args.lr_backbone_ratio
    param_dict = [
        {'params': [p for n, p in model_without_ddp.named_parameters() if 'backbone' not in n and p.requires_grad],
         'lr': args.lr},
        {'params': [p for n, p in model_without_ddp.named_parameters() if 'backbone' in n and p.requires_grad],
         'lr': backbone_lr},
    ]
    
    optimizer = torch.optim.AdamW(param_dict, lr=args.lr, weight_decay=args.weight_decay)

    return optimizer

def build_lr_scheduler(optimizer, last_epoch, args):
    return LinearDecayLR(optimizer, args, last_epoch=last_epoch)