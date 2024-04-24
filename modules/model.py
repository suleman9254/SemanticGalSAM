
from tqdm import tqdm
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from modules.build_sam import build_sam
from modules.lora import build_lora

from statistics import mean

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
        
        self.model = nn.DataParallel(self.model)
        
    def forward(self, pixel_values, output_shape):
        outputs = self.model(pixel_values=pixel_values, multimask_output=True)
        logit = torch.squeeze(outputs.pred_masks, dim=1)
        logit = F.interpolate(logit, output_shape, mode='bilinear', align_corners=False)
        return logit
    
    def fit(self, cfg):
        self._configureDevice(cfg['device'])
        self._configureOptimizer(cfg['lr'])
        self._configureMetric(cfg['metric'])
        self.model.to(self.device)

        bestScores = {'Loss': 10000}
        with tqdm(range(cfg['epochs']), desc='Training') as tepoch:
            for epoch in tepoch:
                self.model.train(True)
                tScores = self.epoch(cfg['trainloader'], update=True)

                print(f'\nTrain Metrics: {tScores}.')

                self.model.eval()
                with torch.no_grad():
                    vScores = self.epoch(cfg['valloader'], update=False)

                print(f'\nValidation Metrics: {vScores}.')

                if vScores['Loss'] < bestScores['Loss']:
                    self.save(cfg['save_path'])
                    bestScores = vScores

                if cfg['wandb']:
                    wandb.log({'train': tScores, 'val': vScores, 'epoch': epoch})
            
                tepoch.set_postfix(tLoss=tScores['Loss'], vLoss=vScores['Loss'])

        self.load(cfg['save_path'])
        return bestScores

    def epoch(self, dataloader, update=False):
        loss_hist = []
        self.metrics.reset()
        
        with tqdm(dataloader, desc='Epoch', leave=False) as tepoch:
            for image, true_mask in tepoch:
                image = image.to(self.device)
                true_mask = true_mask.to(self.device)

                _, h, w = true_mask.shape
                logit = self.forward(image, output_shape=(h,w))
                
                loss = self.loss(logit, true_mask)
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
        loss.backward()
        self.optimizer.step()
        return None

    def _configureOptimizer(self, lr):
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(params, lr=lr)
        self.loss = nn.CrossEntropyLoss(weight=torch.tensor([1, 3, 3, 3, 3], dtype=torch.float))
        return None
    
    def _configureDevice(self, device):
        self.device = device
        return None
    
    def _configureMetric(self, metrics):
        self.metrics = metrics
        return None

    def save(self, path):
        state = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        return torch.save(state, path)

    def load(self, path):
        state = torch.load(path)
        return self.model.load_state_dict(state, strict=False)