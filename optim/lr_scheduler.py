import warnings
from torch.optim.lr_scheduler import _LRScheduler

class LinearDecayLR(_LRScheduler):
    def __init__(self, optimizer, args, last_epoch=-1, verbose=False):
        if args.finetune:
            self.lrs = [args.lr] * (args.epochs + 1)
        else:
            warmup_lr = [args.warmup_min_lr + ((args.lr - args.warmup_min_lr) * i / args.warmup_epochs) for i in range(args.warmup_epochs)]
            decay_lr = [max(i * args.lr / args.epochs, args.min_lr) for i in range(args.epochs - args.warmup_epochs)]
            decay_lr.reverse()
            self.lrs = warmup_lr + decay_lr + decay_lr[-1:]
        
        self.lr_backbone_ratio = args.lr_backbone_ratio
        super(LinearDecayLR, self).__init__(optimizer, last_epoch, verbose)
    
    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                        "please use `get_last_lr()`.", UserWarning) 

        lr = self.lrs[self.last_epoch]
        return [lr, lr * self.lr_backbone_ratio]
    
    

