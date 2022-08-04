import torch
import random
import numpy as np
from utils.nested_tensor import nested_tensor_from_tensor_list
from utils.misc import sample_bezier_curve

class SeqConstructor(object):
    def __init__(self, args):
        self.num_bins = args.num_bins 
        self.max_num_text_ins = args.max_num_text_ins
        self.sos_index = args.sos_index
        self.eos_index = args.eos_index
        self.padding_index = args.padding_index

    def __call__(self, targets):
        input_seqs_ = []
        output_seqs_ = []

        for target in targets:    
            if target['bezier_pts'].shape[0] > self.max_num_text_ins:
                keep = random.sample(range(target['bezier_pts'].shape[0]), self.max_num_text_ins)
                keep = torch.tensor(keep)
                target['bezier_pts'] = target['bezier_pts'][keep]
                target['recog'] = target['recog'][keep]
            
            if target['bezier_pts'].shape[0] > 0:
                center_pts = []
                for bezier_pt in target['bezier_pts']:
                    bezier_pt = bezier_pt.numpy().reshape(8, 2)
                    mid_pt1 = sample_bezier_curve(bezier_pt[:4], mid_point=True)
                    mid_pt2 = sample_bezier_curve(bezier_pt[4:], mid_point=True)
                    center_pt = (mid_pt1 + mid_pt2) / 2
                    center_pts.append(center_pt)
                center_pts = np.concatenate(center_pts)
                center_pts = torch.from_numpy(center_pts).type(torch.float32)
            else:
                center_pts = torch.ones(0).reshape(-1, 2).type(torch.float32)
            center_pts = (center_pts * self.num_bins).floor().type(torch.long)
            center_pts = torch.clamp(center_pts, min=0, max=self.num_bins - 1)

            recog_label = target['recog'] + self.num_bins

            pt_label = torch.cat([center_pts, recog_label], dim=-1)
            idx = torch.randperm(pt_label.shape[0])
            pt_label = pt_label[idx]
            
            pt_label = pt_label.flatten()
            input_seq = torch.cat([torch.tensor([self.sos_index], dtype=torch.long), pt_label])
            output_seq = torch.cat([pt_label, torch.tensor([self.eos_index], dtype=torch.long)])
        
            input_seqs_.append(input_seq)
            output_seqs_.append(output_seq)

        max_seq_length = max([len(seq) for seq in input_seqs_])
        input_seqs = torch.ones(len(input_seqs_), max_seq_length, dtype=torch.long) * self.padding_index
        output_seqs = torch.ones(len(output_seqs_), max_seq_length, dtype=torch.long) * self.padding_index
        for i in range(len(input_seqs_)):
            input_seqs[i, :len(input_seqs_[i])].copy_(input_seqs_[i])
            output_seqs[i, :len(output_seqs_[i])].copy_(output_seqs_[i])
        
        return input_seqs, output_seqs


class CollateFN(object):
    def __init__(self, image_set, args):
        self.seq_constructor = SeqConstructor(args)
        self.batch_aug = args.batch_aug
        self.train = (image_set == 'train')

    def __call__(self, batch):
        batch = list(zip(*batch))

        if self.batch_aug:
            tensors = batch[0] + batch[1]
            targets = batch[2] + batch[3]
        else:
            tensors = batch[0]
            targets = batch[1]
        images = nested_tensor_from_tensor_list(tensors)

        if self.train:
            input_seqs, output_seqs = self.seq_constructor(targets)
            return images, input_seqs, output_seqs 
        else:
            return images, targets



