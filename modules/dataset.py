import torch
from torch.utils.data import Dataset
from transformers import SamProcessor, SamImageProcessor

import sys
sys.path.append('/home/msuleman/ml20_scratch/fyp_galaxy')
from base_dataset import RadioGalaxyNET

class SAMDataset(Dataset):
    def __init__(self, root, annFile, image_size, transform=None, transforms=None):
        self.dataset = RadioGalaxyNET(root, annFile, transform, transforms)

        self.processor = SamImageProcessor(size={"longest_edge": image_size}, 
                                           pad_size={"height": image_size, 
                                                     "width": image_size})
        self.processor = SamProcessor(self.processor)

    def __getitem__(self, idx):
        img, ann = self.dataset[idx]
        target = self.__makeTarget__(ann)
        input = self.processor(img, return_tensors="pt")
        pixel_values = torch.squeeze(input['pixel_values'])
        return pixel_values, target
        
    def __makeTarget__(self, ann):
        masks, labels = ann['masks'], ann['labels']

        _, h, w = masks.shape # anticipating earlier transforms
        target = torch.zeros((h, w), dtype=torch.long)
        for m, c in zip(masks, labels):
            target[m == 1] = c
        return target

    def __len__(self):
        return len(self.dataset)