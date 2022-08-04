import torch

class Checkpointer(object):
    def __init__(self, distributed):
        self.distributed = distributed

    def load(self, checkpoint_path, model, optimizer=None):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        if self.distributed:
            model = model.module
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
        
        if (not optimizer is None) and ('optimizer' in checkpoint):
            optimizer.load_state_dict(checkpoint['optimizer']) 
        
        if 'epoch' in checkpoint:
            last_epoch = checkpoint['epoch']
        else:
            last_epoch = -1
        
        return last_epoch
    
    def save(self, checkpoint_path, model, optimizer, epoch):
        if self.distributed:
            model = model.module
        
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
        }
        torch.save(checkpoint, checkpoint_path)