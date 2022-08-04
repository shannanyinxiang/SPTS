import sys 
sys.path.append('.')

import os
import cv2 
import random
from tqdm import tqdm
from dataset import build_dataset
from utils.visualize import visualize_annotations

def main(args):
    dataset = build_dataset(args.image_set, args)

    if args.random_vis:
        indices = random.sample(list(range(len(dataset))), args.vis_num_samples)
    else:
        indices = range(len(dataset))
    
    os.makedirs(args.output_folder, exist_ok=True)
    for idx in tqdm(indices):
        image, target = dataset[idx]

        image_bboxes, image_bezier_curves, image_texts = \
            visualize_annotations(image, target, args.chars)
        
        file_name = target['file_name'].split('.')[0]
        cv2.imwrite(os.path.join(args.output_folder, f'{file_name}_bbox.jpg'), image_bboxes)
        cv2.imwrite(os.path.join(args.output_folder, f'{file_name}_bezier.jpg'), image_bezier_curves)
        cv2.imwrite(os.path.join(args.output_folder, f'{file_name}_text.jpg'), image_texts)

if __name__ == '__main__':
    from utils.parser import DefaultParser
    
    parser = DefaultParser()
    parser.add_argument('--image_set', type=str)
    parser.add_argument('--random_vis', action='store_true')
    parser.add_argument('--vis_num_sample', type=int)
    args = parser.parse_args()

    main(args)
        