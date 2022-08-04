import sys 
sys.path.append('.')

import os 
import cv2
from tqdm import tqdm
from dataset import build_dataset, build_dataloader
from utils.visualize import visualize_seq

def main(args):
    dataset = build_dataset(args.image_set, args)
    dataloader, _ = build_dataloader(dataset, args.image_set, args)

    os.makedirs(args.output_folder, exist_ok=True)
    count = 0
    for images, input_seqs, output_seqs in tqdm(dataloader):
        images = images.unpad_tensors()
        for image, input_seq, output_seq in zip(images, input_seqs, output_seqs):
            input_seq = input_seq[input_seq != args.padding_index]
            output_seq = output_seq[output_seq != args.padding_index]
            assert(input_seq[1:].equal(output_seq[:-1]))

            image = visualize_seq(image, input_seq, 'input', args)
            save_path = os.path.join(args.output_folder, f'{count:08d}.jpg')
            cv2.imwrite(save_path, image)

            count += 1    

if __name__ == '__main__':
    from utils.parser import DefaultParser

    parser = DefaultParser()
    parser.add_argument('--image_set', type=str)
    args = parser.parse_args()
    args.distributed = False

    main(args)