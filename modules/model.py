
import wandb
import pandas as pd
from tqdm import tqdm
from statistics import mean

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from monai.losses import GeneralizedDiceLoss

from modules.build_sam import build_sam
from modules.lora import build_lora
from modules.lr_scheduler import Warmup
from torch.optim.lr_scheduler import ReduceLROnPlateau

def freeze_backbone(model):
    for name, param in model.named_parameters():
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
            param.requires_grad_(False)
    return model

class SAM(nn.Module):
    def __init__(self, 
                 pretrained_path, 
                 num_classes, 
                 image_size, 
                 vit_patch_size, 
                 lora_regex, 
                 normal_regex,
                 lora_rank, 
                 lora_alpha):
        
        super().__init__()
        self.model = build_sam(pretrained_path, num_classes, image_size, vit_patch_size)

        if lora_regex:
            self.model = build_lora(self.model, lora_regex, normal_regex, lora_rank, lora_alpha)
        else:
            self.model = freeze_backbone(self.model)
        
    def forward(self, pixel_values, output_shape):
        outputs = self.model(pixel_values=pixel_values, multimask_output=True)
        logit = torch.squeeze(outputs.pred_masks, dim=1)
        logit = F.interpolate(logit, output_shape, mode='bilinear', align_corners=False)
        return logit
    
    def fit(self, cfg):
        self.trainloader, self.valloader = cfg['trainloader'], cfg['valloader']
        self.accelerator, self.metrics = cfg['accelerator'], cfg['metric']
        self.monitored_metric = cfg['monitored_metric']
        
        self._configureOptimizer(lr=cfg['lr'], 
                                 weight_decay=cfg['weight_decay'], 
                                 patience=cfg['scheduler_patience'], 
                                 threshold=cfg['scheduler_threshold'], 
                                 lambda_dice=cfg['lambda_dice'])
        
        self.model.to(self.accelerator.device)
        self._configureAccelerator(self.accelerator)

        early_stop, bestMetric = 0, 0
        bestScores = {'Loss': 10000}
        with tqdm(range(cfg['epochs']), desc='Training') as tepoch:
            for epoch in tepoch:
                self.model.train(True)
                tScores = self.epoch(self.trainloader, update=True)

                print('\n-----------------------------Train Metrics-----------------------------')
                print(pd.DataFrame(tScores, index=['']))
                print('-------------------------------------------------------------------------')

                self.model.eval()
                with torch.no_grad():
                    vScores = self.epoch(self.valloader, update=False)

                print('-----------------------------Val Metrics-----------------------------')
                print(pd.DataFrame(vScores, index=['']))
                print('---------------------------------------------------------------------')

                if cfg['wandb']:
                    wandb.log({'train': tScores, 'val': vScores, 'epoch': epoch})

                if vScores['Loss'] < bestScores['Loss']:
                    unwrapped_model = self.accelerator.unwrap_model(self.model)
                    self.save(cfg['save_path'], unwrapped_model)
                    bestScores = vScores

                if vScores[self.monitored_metric] < bestMetric + cfg['early_stopping_threshold']:
                    early_stop = early_stop + 1
                    
                    if early_stop > cfg['early_stopping_patience']:
                        break
                    
                else:
                    early_stop, bestMetric = 0, vScores[self.monitored_metric]

                self.scheduler.step(vScores[self.monitored_metric])
                tepoch.set_postfix(tLoss=tScores['Loss'], vLoss=vScores['Loss'])

        self.model = self.accelerator.unwrap_model(self.model)
        self.load(cfg['save_path'])
        return bestScores

    def epoch(self, dataloader, update=False):
        loss_hist = []
        self.metrics.reset()
        
        with tqdm(dataloader, desc='Epoch', leave=False) as tepoch:
            for image, true_mask in tepoch:
                with self.accelerator.accumulate(self.model):
                    _, h, w = true_mask.shape
                    logit = self.forward(image, output_shape=(h,w))

                    loss = self.loss(logit, torch.unsqueeze(true_mask, dim=1))
                    if update:
                        self._updateWeights(loss)
                    
                    loss_hist.append(loss.item())
                    self.metrics.update(logit.detach().cpu(), true_mask.cpu())
                    
                    tepoch.set_postfix(loss = mean(loss_hist))

        scores = self.metrics.compute()
        scores['Loss'] = mean(loss_hist)
        return scores
    
    def _updateWeights(self, loss):
        self.optimizer.zero_grad()
        self.accelerator.backward(loss)
        self.optimizer.step()
        return None

    def _configureOptimizer(self, lr, weight_decay, patience, threshold, lambda_dice):
        params = [p for p in self.model.parameters() if p.requires_grad]    
        self.optimizer = AdamW(params, lr=lr, weight_decay=weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', patience=patience, threshold=threshold)
        self.loss = GeneralizedDiceLoss(to_onehot_y=True, softmax=True)
        return None
    
    def _configureAccelerator(self, accelerator):
        self.model, self.optimizer, self.trainloader, self.valloader, self.scheduler = accelerator.prepare(self.model, self.optimizer, self.trainloader, self.valloader, self.scheduler)
        return None

    def save(self, path, model=None):
        model = self.model if not model else model
        state = {n: p for n, p in model.named_parameters() if p.requires_grad}
        return torch.save(state, path)

    def load(self, path):
        state = torch.load(path)
        return self.model.load_state_dict(state, strict=False)