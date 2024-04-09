import numpy as np
import random
import torch

import optuna

from torch.utils.data import DataLoader
from modules.dataset import SAMDataset
from modules.model import SAM
from modules.utils import mask_decoder_regex

from torchmetrics import Accuracy, JaccardIndex, MetricCollection

def reset_seed(n):
    np.random.seed(n)
    torch.manual_seed(n)
    random.seed(n)

batch_size = 10
lora_regex = mask_decoder_regex
pretrained_path = 'facebook/sam-vit-base'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

root, annFile = '../data/train', '../data/annotations/train.json'
trainset = SAMDataset(root, annFile, pretrained_path)

root, annFile = '../data/val', '../data/annotations/val.json'
valset = SAMDataset(root, annFile, pretrained_path)

metric = MetricCollection([Accuracy(task='multiclass', num_classes=5), 
                           JaccardIndex(task='multiclass', num_classes=5)])

def objective(trial):
    
    reset_seed(n=42)

    lr = trial.suggest_float("lr", low=5e-5, high=5e-3, log=True)
    epochs = trial.suggent_int("epochs", low=5, high=50, step=5)
    lora_rank = trial.suggent_int("lora_rank", low=5, high=50, step=5)
    lora_alpha = trial.suggent_int("lora_alpha", low=5, high=50, step=5)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)

    model = SAM(pretrained_path, num_classes=5, lora_regex=lora_regex, lora_rank=lora_rank, lora_alpha=lora_alpha)
    
    save_path = f'saves/lr_{lr}_epochs_{epochs}_lora_rank_{lora_rank}_lora_alpha_{lora_alpha}_batch_size_{batch_size}_mask_decoder.ckpt'

    cfg = {'trainloader': trainloader, 
       'valloader': valloader, 
       'epochs': epochs, 
       'lr': lr,
       'save_path': save_path, 
       'device': device, 
       'metric': metric}

    vScores = model.fit(cfg)
    return vScores 

study = optuna.create_study(study_name="mask_decoder", 
                            direction='minimize', 
                            load_if_exists=True, 
                            storage="sqlite:///mask_decoder.db")

study.optimize(objective, n_trials=10)