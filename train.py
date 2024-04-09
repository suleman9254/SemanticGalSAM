import sys
sys.path.append('/home/msuleman/ml20_scratch/fyp_galaxy')

import numpy as np
import random
import torch

from torch.utils.data import DataLoader
from modules.dataset import SAMDataset
from modules.model import SAM
from modules.utils import mask_decoder_regex, vision_encoder_regex, auto_save_path

from torchmetrics import Accuracy, JaccardIndex, MetricCollection
from torchmetrics.detection import MeanAveragePrecision

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--lr', type=float, required=True)
parser.add_argument('--epochs', type=int, required=True)
parser.add_argument('--lora_rank', type=int, required=True)
parser.add_argument('--lora_alpha', type=float, required=True)
parser.add_argument('--batch_size', type=int, required=True)
args = parser.parse_args()

def reset_seed(n):
    np.random.seed(n)
    torch.manual_seed(n)
    random.seed(n)

reset_seed(n=42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pretrained_path = 'facebook/sam-vit-base'

lora_regex = vision_encoder_regex
model = SAM(pretrained_path, num_classes=5, lora_regex=lora_regex, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha)

root, annFile = '../data/train', '../data/annotations/train.json'
trainset = SAMDataset(root, annFile, pretrained_path)
trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)

root, annFile = '../data/val', '../data/annotations/val.json'
valset = SAMDataset(root, annFile, pretrained_path)
valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=False)

metric = MetricCollection([Accuracy(task='multiclass', num_classes=5), 
                           JaccardIndex(task='multiclass', num_classes=5)])

save_path = auto_save_path(args)

print(save_path)

cfg = {'trainloader': trainloader, 
       'valloader': valloader, 
       'epochs': args.epochs, 
       'lr': args.lr,
       'save_path': save_path, 
       'device': device, 
       'metric': metric}

vScores = model.fit(cfg)

print(vScores)