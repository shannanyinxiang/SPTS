import sys
sys.path.append('.')

import os
import json
import time
import torch
import random 
import datetime
import numpy as np

from model import build_model
from engine import train_one_epoch, validate
from dataset import build_dataset, build_dataloader
from optim import build_criterion, build_lr_scheduler, build_optimizer
from utils.misc import get_sha
from utils.checkpointer import Checkpointer
from utils.dist import init_distributed_mode, get_rank, is_main_process


def main(args):
    init_distributed_mode(args)

    print(f'git:\n {get_sha()}\n')
    print(args)

    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = build_model(args)     
    n_parameters = sum(p.numel() for p in model.parameters())
    print('Number of params:', n_parameters)   

    checkpointer = Checkpointer(args.distributed)
    
    if args.eval:
        val_dataset = build_dataset('val', args)
        val_dataloader, _ = build_dataloader(val_dataset, 'val', args)
        assert(args.resume != '')
        epoch = checkpointer.load(args.resume, model)
        validate(model, val_dataloader, epoch, args)
        return 

    criterion = build_criterion(args)
    optimizer = build_optimizer(model, args)
    if args.resume != '':
        last_epoch = checkpointer.load(args.resume, model, optimizer)
    else:
        last_epoch = -1
    lr_scheduler = build_lr_scheduler(optimizer, last_epoch, args)

    train_dataset = build_dataset('train', args)
    train_dataloader, train_sampler = build_dataloader(train_dataset, 'train', args)
    
    checkpoint_folder = os.path.join(args.output_folder, 'checkpoints')
    os.makedirs(checkpoint_folder, exist_ok=True)

    print('Start training')
    start_time = time.time()
    for epoch in range(last_epoch+1, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        train_stats = train_one_epoch(model, train_dataloader, criterion, optimizer, epoch, args)

        lr_scheduler.step()

        if (epoch + 1) % args.checkpoint_freq == 0:
            checkpoint_name = f'checkpoint_ep{epoch:04}.pth'
            checkpoint_path = os.path.join(checkpoint_folder, checkpoint_name)
            save_dict = {
                'model': model.module.state_dict() if args.distributed else model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'args': args,
            }
            if is_main_process():
                torch.save(save_dict, checkpoint_path)

        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            'epoch': epoch,
            'n_parameters': n_parameters,
        }
        if is_main_process():
            with open(os.path.join(args.output_folder, 'log.txt'), 'a') as f:
                f.write(json.dumps(log_stats) + '\n')

    total_time = time.time() - start_time 
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'Training time {total_time_str}')

if __name__ == '__main__':
    from utils.parser import DefaultParser

    parser = DefaultParser()
    args = parser.parse_args()

    main(args)