from .resnet import ResNet
from .joiner import Joiner
from .position_embedding import PositionEmbeddingSine, PositionEmbeddingLearned

def build_backbone(args):
    position_embedding = build_position_embedding(args)

    train_backbone = args.lr_backbone_ratio > 0
    if 'resnet' in args.backbone:
        backbone = ResNet(
            name=args.backbone,
            train_backbone=train_backbone,
            return_interm_layers=False,
            dilation=False,
            freeze_bn=args.freeze_bn
        )
    else:
        raise NotImplementedError

    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model

def build_position_embedding(args):
    N_steps = args.tfm_hidden_dim // 2
    if args.position_embedding in ('v2', 'sine'):
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif args.position_embedding in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {args.position_embedding}")

    return position_embedding