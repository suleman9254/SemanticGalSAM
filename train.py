import sys
sys.path.append('/home/msuleman/ml20_scratch/fyp_galaxy')

import numpy as np
import random
import torch

from torch.utils.data import DataLoader
from modules.dataset import SAMDataset
from modules.model import (SAM, 
                           mask_decoder_regex, 
                           vision_encoder_regex, 
                           mask_decover_vision_encoder_regex)

from torchmetrics import Accuracy, JaccardIndex, MetricCollection
from torchmetrics.detection import MeanAveragePrecision

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--lr', type=float, required=True)
parser.add_argument('--epochs', type=int, required=True)
parser.add_argument('--save_path', type=str, required=True)
parser.add_argument('--batch_size', type=int, required=True)
args = parser.parse_args()

def reset_seed(n):
    np.random.seed(n)
    torch.manual_seed(n)
    random.seed(n)

reset_seed(n=42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pretrained_path = 'facebook/sam-vit-base'

model = SAM(pretrained_path, num_classes=5, lora_regex=mask_decoder_regex, lora_rank=60)

root, annFile = '../data/train', '../data/annotations/train.json'
trainset = SAMDataset(root, annFile, pretrained_path)
trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)

root, annFile = '../data/val', '../data/annotations/val.json'
valset = SAMDataset(root, annFile, pretrained_path)
valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=False)

metric = MetricCollection([Accuracy(task='multiclass', num_classes=5), 
                           JaccardIndex(task='multiclass', num_classes=5)])

cfg = {'trainloader': trainloader, 
       'valloader': valloader, 
       'epochs': args.epochs, 
       'lr': args.lr,
       'save_path': args.save_path, 
       'device': device, 
       'metric': metric}

vLoss, vScores = model.fit(cfg)

print(vLoss)
print(vScores)