import torch
import wandb

from torch.utils.data import DataLoader
from modules.dataset import SAMDataset
from modules.model import SAM
from modules.lora import fetch_lora_regex
from modules.utils import reset_seed, seed_worker, generator

from torchmetrics import Accuracy, F1Score, JaccardIndex, MetricCollection

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('-l', '--lr', type=float, required=True)
parser.add_argument('-e', '--epochs', type=int, required=True)
parser.add_argument('-b', '--batch_size', type=int, required=True)

parser.add_argument('-ll', '--lora_layers', type=str, default=None)
parser.add_argument('-lr', '--lora_rank', type=int, default=None)
parser.add_argument('-la', '--lora_alpha', type=float, default=None)

parser.add_argument('-s', '--savename', type=str, required=True)
parser.add_argument('-w', '--wandb', action="store_true")
args = parser.parse_args()

reset_seed(n=42)
generator.manual_seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if args.wandb:
    wandb.login()
    wandb.init(project='SemanticSAM', 
               config=vars(args), 
               name=args.savename)

num_classes = 5
image_size = 512
vit_patch_size = 8
pretrained_path = 'facebook/sam-vit-base'
save_path = f'saves/{args.savename}.ckpt'
lora_regex, normal_layers = fetch_lora_regex(args.lora_layers)

model = SAM(pretrained_path, 
            num_classes=num_classes, 
            image_size=image_size, 
            vit_patch_size=vit_patch_size, 
            lora_regex=lora_regex, 
            normal_regex=normal_layers, 
            lora_rank=args.lora_rank, 
            lora_alpha=args.lora_alpha)

root, annFile = '../data/train', '../data/annotations/train.json'
trainset = SAMDataset(root, annFile, image_size, means=[-1.8163, -1.9570, -1.7297], stds=[0.8139, 0.4834, 0.4621])
trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, worker_init_fn=seed_worker, generator=generator)

root, annFile = '../data/val', '../data/annotations/val.json'
valset = SAMDataset(root, annFile, image_size, means=[-1.8168, -1.9573, -1.7303], stds=[0.8127, 0.4824, 0.4602])
valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, worker_init_fn=seed_worker, generator=generator)

metric = MetricCollection([Accuracy(task='multiclass', num_classes=num_classes, average='macro'), 
                           F1Score(task="multiclass", num_classes=num_classes, average='macro'),
                           JaccardIndex(task='multiclass', num_classes=num_classes, average='macro')])

cfg = {'trainloader': trainloader, 
       'valloader': valloader, 
       'epochs': args.epochs, 
       'lr': args.lr,
       'save_path': save_path, 
       'device': device, 
       'metric': metric, 
       'wandb': args.wandb}

vScores = model.fit(cfg)

print(vScores)