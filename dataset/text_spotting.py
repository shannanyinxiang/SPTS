import torch
import torchvision
from copy import deepcopy
from utils.misc import bezier2bbox

class TextSpottingDataset(torchvision.datasets.CocoDetection):
    def __init__(self, image_folder, anno_file, dataset_name, batch_aug=True, transforms=None):
        super(TextSpottingDataset, self).__init__(image_folder, anno_file)
        self.dataset_name = dataset_name 
        self.image_folder = image_folder
        self.batch_aug = batch_aug if transforms is not None else False
        self._transforms = transforms 

    def __getitem__(self, index):
        image, anno = super(TextSpottingDataset, self).__getitem__(index)

        image_w, image_h = image.size
        anno = [ele for ele in anno if 'iscrowd' not in anno or ele['iscrowd'] == 0]

        target = {}
        target['image_id'] = self.ids[index]
        target['file_name'] = self.coco.loadImgs(self.ids[index])[0]['file_name']
        target['image_folder'] = self.image_folder
        target['dataset_name'] = self.dataset_name

        classes = [ele['category_id'] for ele in anno]
        classes = torch.tensor(classes, dtype=torch.int64)
        target['labels'] = classes

        area = torch.tensor([ele['area'] for ele in anno], dtype=torch.float32)
        target['area'] = area 

        iscrowd = torch.tensor([ele['iscrowd'] if 'iscrowd' in ele else 0 for ele in anno])
        target['iscrowd'] = iscrowd

        image_size = torch.tensor([int(image_h), int(image_w)])
        target['orig_size'] = image_size 
        target['size'] = image_size 

        recog = [ele['rec'] for ele in anno]
        recog = torch.tensor(recog, dtype=torch.long).reshape(-1, 25)
        target['recog'] = recog 

        bezier_pts = [ele['bezier_pts'] for ele in anno]
        bezier_pts = torch.tensor(bezier_pts, dtype=torch.float32).reshape(-1, 16)
        target['bezier_pts'] = bezier_pts

        bboxes = []
        for bezier_pt in bezier_pts:
            bezier_pt = bezier_pt.numpy()
            bbox = bezier2bbox(bezier_pt)
            bboxes.append(bbox)
        bboxes = torch.tensor(bboxes, dtype=torch.float32).reshape(-1, 4)
        target['bboxes'] = bboxes

        if self.batch_aug:
            image1, target1 = self._transforms(deepcopy(image), deepcopy(target))
            image2, target2 = self._transforms(deepcopy(image), deepcopy(target))
        else:
            if not self._transforms is None:
                image, target = self._transforms(image, target)

        if self.batch_aug:
            return image1, image2, target1, target2
        else:
            return image, target 