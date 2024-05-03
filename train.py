import wandb

from torch.utils.data import DataLoader
from modules.dataset import SAMDataset
from modules.model import SAM
from modules.lora import fetch_lora_regex
from modules.utils import reset_seed, seed_worker, generator

from accelerate import Accelerator
from torchvision.transforms import v2
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
accelerator = Accelerator()

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

weight_decay = 0.015346716328972766
scheduler_patience = 5
scheduler_threshold = 0.01
early_stopping_patience = 10
early_stopping_threshold = 0.01
lambda_dice = 0.4187426704271407

monitored_metric = 'MulticlassJaccardIndex'

model = SAM(pretrained_path, 
            num_classes=num_classes, 
            image_size=image_size, 
            vit_patch_size=vit_patch_size, 
            lora_regex=lora_regex, 
            normal_regex=normal_layers, 
            lora_rank=args.lora_rank, 
            lora_alpha=args.lora_alpha)

transforms = v2.Compose([v2.RandomRotation(degrees=180)])
root, annFile = '../data/train', '../data/annotations/train.json'
trainset = SAMDataset(root, annFile, image_size, transforms=transforms)
trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, worker_init_fn=seed_worker, generator=generator)

root, annFile = '../data/val', '../data/annotations/val.json'
valset = SAMDataset(root, annFile, image_size)
valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, worker_init_fn=seed_worker, generator=generator)

metric = MetricCollection([Accuracy(task='multiclass', num_classes=num_classes, average='macro'), 
                           F1Score(task="multiclass", num_classes=num_classes, average='macro'),
                           JaccardIndex(task='multiclass', num_classes=num_classes, average='macro')])

cfg = {'trainloader': trainloader, 
       'valloader': valloader, 
       'epochs': args.epochs, 
       'lr': args.lr,
       'save_path': save_path, 
       'accelerator': accelerator, 
       'metric': metric, 
       'weight_decay': weight_decay, 
       'scheduler_patience': scheduler_patience, 
       'scheduler_threshold': scheduler_threshold, 
       'lambda_dice': lambda_dice, 
       'early_stopping_threshold': early_stopping_threshold, 
       'early_stopping_patience': early_stopping_patience, 
       'monitored_metric': monitored_metric,
       'wandb': args.wandb}

vScores = model.fit(cfg)

print(vScores)