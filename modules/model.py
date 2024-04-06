
import re as regex

import torch.nn as nn
import torch.nn.functional as F

from peft import LoraConfig, get_peft_model
from transformers.models.sam.modeling_sam import SamModel, SamMaskDecoder

mask_decoder_regex = r'^mask_decoder\.transformer.*(?:self_attn\.(v|q)_proj|cross_attn_token_to_image\.(q|v)_proj|cross_attn_image_to_token\.(q|v)_proj|final_attn_token_to_image\.(q|v)_proj).*'
vision_encoder_regex = r'^vision_encoder\.layers.*(attn\.qkv).*'
mask_decover_vision_encoder_regex = regex.compile(f"{vision_encoder_regex}|{mask_decoder_regex}")
prediction_head_regex = r'^mask_decoder.*(?:iou_prediction_head|output_hypernetworks_mlps|mask_tokens).*'

def replace_decoder(model, num_classes):
    mask_decoder_config = model.config.mask_decoder_config.to_dict()
    mask_decoder_config['num_multimask_outputs'] = num_classes
    model.config.mask_decoder_config.update(mask_decoder_config)
    
    # preserve weights
    custom_decoder = SamMaskDecoder(model.config.mask_decoder_config)
    custom_decoder.transformer = model.mask_decoder.transformer
    custom_decoder.upscale_conv1 = model.mask_decoder.upscale_conv1
    custom_decoder.upscale_conv2 = model.mask_decoder.upscale_conv2
    custom_decoder.upscale_layer_norm = model.mask_decoder.upscale_layer_norm
    
    model.mask_decoder = custom_decoder
    return model

def get_module_list(model, reg_exp):
    target_modules = set()
    for name, _ in model.named_parameters():
        if regex.search(reg_exp, name):
            codeword = regex.sub(r'\.(weight|bias)$', '', name)
            target_modules.add(codeword)
    
    return target_modules

def get_lora(model, lora_regex, lora_rank):            
    target_modules = get_module_list(model, lora_regex)
    modules_to_save = get_module_list(model, prediction_head_regex)

    config = LoraConfig(r=lora_rank, 
                        modules_to_save=modules_to_save, 
                        target_modules=target_modules)
    
    return get_peft_model(model, config)

class SAM(nn.Module):
    def __init__(self, pretrained_path, num_classes, lora_regex, lora_rank):
        self.model = SamModel.from_pretrained(pretrained_path)
        self.model = replace_decoder(self.model, num_classes)
        self.model = get_lora(self.model, lora_regex, lora_rank)
        
        self.loss = nn.CrossEntropyLoss(ignore_index=0)
        
    def forward(self, pixel_values, output_shape):
        outputs = self.model(pixel_values=pixel_values, multimask_output=True)
        logit = F.interpolate(outputs.pred_masks, output_shape, mode='bilinear', align_corners=False)
        return logit

    
    
    
    
        
                                           
