import re as regex
from peft import LoraConfig, get_peft_model

def fetch_lora_regex(training_scheme):
    vision_encoder_regex = r'^vision_encoder\.layers.*(attn\.qkv).*'
    prediction_head_regex = r'^mask_decoder.*(?:iou_prediction_head|output_hypernetworks_mlps|mask_tokens).*'
    mask_decoder_regex = r'^mask_decoder\.transformer.*(?:self_attn\.(v|q)_proj|cross_attn_token_to_image\.(q|v)_proj|cross_attn_image_to_token\.(q|v)_proj|final_attn_token_to_image\.(q|v)_proj).*'

    if training_scheme == 'vision_encoder':
        return vision_encoder_regex, prediction_head_regex
    elif training_scheme == 'vision_encoder_mask_decoder':
        return [vision_encoder_regex, mask_decoder_regex], prediction_head_regex
    else:
        return None, None

def join_regex(reg_exp_list):
    reg_exp = '|'.join(reg_exp_list)
    return regex.compile(reg_exp)

def regex_parameter_search(model, reg_exp):
    if isinstance(reg_exp, list):
        reg_exp = join_regex(reg_exp)

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