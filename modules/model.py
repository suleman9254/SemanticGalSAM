
from tqdm import tqdm
import re as regex

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from peft import LoraConfig, get_peft_model
from transformers.models.sam.modeling_sam import SamModel, SamMaskDecoder

from modules.utils import regex_parameter_search, prediction_head_regex

def replace_decoder(model, num_classes):
    mask_decoder_config = model.config.mask_decoder_config.to_dict()
    mask_decoder_config['num_multimask_outputs'] = num_classes
    model.config.mask_decoder_config.update(mask_decoder_config)
    
    # preserve some weights
    custom_decoder = SamMaskDecoder(model.config.mask_decoder_config)
    custom_decoder.transformer = model.mask_decoder.transformer
    custom_decoder.upscale_conv1 = model.mask_decoder.upscale_conv1
    custom_decoder.upscale_conv2 = model.mask_decoder.upscale_conv2
    custom_decoder.upscale_layer_norm = model.mask_decoder.upscale_layer_norm
    
    model.mask_decoder = custom_decoder
    return model

def make_lora(model, lora_regex, lora_rank, lora_alpha):            
    target_modules = regex_parameter_search(model, lora_regex)
    modules_to_save = regex_parameter_search(model, prediction_head_regex)

    config = LoraConfig(r=lora_rank, 
                        lora_alpha=lora_alpha,
                        modules_to_save=modules_to_save, 
                        target_modules=target_modules)
    
    return get_peft_model(model, config)

class SAM(nn.Module):
    def __init__(self, pretrained_path, num_classes, lora_regex, lora_rank, lora_alpha):
        super().__init__()
        self.model = SamModel.from_pretrained(pretrained_path)
        self.model = replace_decoder(self.model, num_classes)
        self.model = make_lora(self.model, lora_regex, lora_rank, lora_alpha)
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

        bestScores = {'loss': 10000}
        with tqdm(range(cfg['epochs']), desc='Training') as tepoch:
            for epoch in tepoch:
                print(f'\nEpoch {epoch}')

                self.model.train(True)
                tScores = self.epoch(cfg['trainloader'], update=True)

                print('Train Metrics:')
                print(tScores)

                self.model.eval()
                with torch.no_grad():
                    vScores = self.epoch(cfg['valloader'], update=False)

                print('Validation Metrics:')
                print(tScores)

                if vScores['loss'] < bestScores['loss']:
                    self.save(cfg['save_path'])
                    bestScores = vScores
            
                tepoch.set_postfix(tLoss=tScores['loss'], vLoss=vScores['loss'])

        self.load(cfg['save_path'])
        return bestScores

    def epoch(self, dataloader, update=False):
        mean_loss = 0
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
                
                mean_loss += loss.item()
                self.metrics.update(logit.detach().cpu(), true_mask.cpu())

                tepoch.set_postfix(batch_loss = loss.item())

        scores = self.metrics.compute()
        scores['loss'] = mean_loss / len(dataloader)
        return scores
    
    def _updateWeights(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return None

    def _configureOptimizer(self, lr):
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(params, lr=lr)
        self.loss = nn.CrossEntropyLoss()
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