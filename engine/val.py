import os
import cv2
import json
import torch
from tqdm import tqdm

from utils.misc import decode_seq
from utils.visualize import visualize_decoded_result

@torch.no_grad()
def validate(model, dataloader, epoch, args):
    model.eval()
    device = torch.device('cuda')
    output_folder = os.path.join(args.output_folder, 'results', f'ep{epoch:03d}')

    results = []
    for samples, targets in tqdm(dataloader):
        assert(len(targets) == 1) # Only support inference with batch size = 1

        samples = samples.to(device)
        seq = torch.ones(1, 1, dtype=torch.long).to(device) * args.sos_index
        output, prob = model(samples, seq)
        output = output[0].cpu()
        prob = prob[0].cpu()

        result = decode_pred_seq(output, prob, targets[0], args)
        results.extend(result)
        
        if args.visualize:
            image = cv2.imread(os.path.join(targets[0]['image_folder'], targets[0]['file_name']))
            image = visualize_decoded_result(image, result)
            save_path = os.path.join(output_folder, 'vis', targets[0]['file_name'])
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, image)

    json_path = os.path.join(output_folder, targets[0]['dataset_name']+'.json')
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, 'w') as f:
        f.write(json.dumps(results, indent=4))


def decode_pred_seq(index_seq, prob_seq, target, args):
    index_seq = index_seq[:-1]
    prob_seq = prob_seq[:-1]
    if len(index_seq) % 27 != 0:
        index_seq = index_seq[:-len(index_seq)%27]
        prob_seq = prob_seq[:-len(prob_seq)%27]
    
    decode_results = decode_seq(index_seq, 'none', args)
    confs = prob_seq.reshape(-1, 27).mean(-1)
    
    image_id = target['image_id']
    image_h, image_w = target['orig_size']
    results = []
    for decode_result, conf in zip(decode_results, confs):
        point_x = decode_result['point'][0] * image_w 
        point_y = decode_result['point'][1] * image_h 
        recog = decode_result['recog']
        result = {
            'image_id': image_id,
            'polys': [[point_x.item(), point_y.item()]],
            'rec': recog,
            'score': conf.item()
        }
        results.append(result)
    
    return results
