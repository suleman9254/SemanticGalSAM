import torch
from torch.utils.data import Dataset
from transformers import SamProcessor

from base_dataset import RadioGalaxyNET

class SAMDataset(Dataset):
    def __init__(self, root, annFile, pretrained_path, transform=None, transforms=None):
        self.dataset = RadioGalaxyNET(root, annFile, transform, transforms)
        self.processor = SamProcessor.from_pretrained(pretrained_path)

    def __getitem__(self, idx):
        img, ann = self.dataset[idx]
        target = self.__makeTarget__(ann)
        input = self.processor(img, return_tensors="pt")
        return input, target
        
    def __makeTarget__(self, ann):
        masks, labels = ann['masks'], ann['labels']

        _, h, w = masks.shape # anticipating earlier transforms
        target = torch.zeros((h, w), dtype=torch.float)
        for m, c in zip(masks, labels):
            target[m == 1] = c
        return target

    def __len__(self):
        return len(self.dataset)