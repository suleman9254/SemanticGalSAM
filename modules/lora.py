import re as regex
from peft import LoraConfig, get_peft_model

def fetch_lora_regex(layers):
    lora_layers = normal_layers = None
    
    if layers == 'vision_encoder':
        lora_layers = r'^vision_encoder\.layers.*(attn\.qkv).*'
        normal_layers = r'^mask_decoder.*?(weight|bias)$'
    
    elif layers == 'vision_encoder_mask_decoder':
        lora_layers = r'^vision_encoder\.layers.*(attn\.qkv).*|^mask_decoder\.transformer.*(?:self_attn\.(v|q)_proj|cross_attn_token_to_image\.(q|v)_proj|cross_attn_image_to_token\.(q|v)_proj|final_attn_token_to_image\.(q|v)_proj).*'
        normal_layers = r'(?![\s\S])'

    return lora_layers, normal_layers

def regex_parameter_search(model, reg_exp):
    target_modules = set()
    for name, _ in model.named_parameters():
        if regex.search(reg_exp, name):
            codeword = regex.sub(r'\.(weight|bias)$', '', name)
            target_modules.add(codeword)
    
    return target_modules

def build_lora(model, lora_layers, normal_layers, lora_rank, lora_alpha):
    target_modules = regex_parameter_search(model, lora_layers)
    modules_to_save = regex_parameter_search(model, normal_layers)

    config = LoraConfig(r=lora_rank, 
                        lora_alpha=lora_alpha,
                        modules_to_save=modules_to_save, 
                        target_modules=target_modules)
    return get_peft_model(model, config)