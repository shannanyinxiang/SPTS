import sys
import math 
import torch
from utils.dist import reduce_dict
from utils.logger import MetricLogger, SmoothedValue

def train_one_epoch(model, dataloader, criterion, optimizer, epoch, args):
    model.train()
    criterion.train()

    metric_logger = MetricLogger(delimiter='  ')    
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'
    device = torch.device('cuda')

    for samples, input_seqs, output_seqs in metric_logger.log_every(dataloader, args.print_freq, header):
        samples = samples.to(device)
        input_seqs = input_seqs.to(device)
        output_seqs = output_seqs.to(device)

        outputs = model(samples, input_seqs)
        ce_loss = criterion(outputs.transpose(1, 2), output_seqs)

        loss_dict = {'ce_loss': ce_loss}
        weight_dict = {'ce_loss': 1}
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        loss_dict_reduced = reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced = sum(loss_dict_reduced_scaled.values()).item()

        if not math.isfinite(losses_reduced):
            print(f'Loss is {losses_reduced}, stopping training')
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if args.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        optimizer.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced_scaled)
        metric_logger.update(lr=optimizer.param_groups[0]['lr'])
    
    metric_logger.synchronize_between_processes()
    print('Averaged stats', metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}