import re as regex

vision_encoder_regex = r'^vision_encoder\.layers.*(attn\.qkv).*'
prediction_head_regex = r'^mask_decoder.*(?:iou_prediction_head|output_hypernetworks_mlps|mask_tokens).*'
mask_decoder_regex = r'^mask_decoder\.transformer.*(?:self_attn\.(v|q)_proj|cross_attn_token_to_image\.(q|v)_proj|cross_attn_image_to_token\.(q|v)_proj|final_attn_token_to_image\.(q|v)_proj).*'

def regex_parameter_search(model, reg_exp):
    if isinstance(reg_exp, list):
        reg_exp = '|'.join(reg_exp)
        reg_exp = regex.compile(reg_exp)

    target_modules = set()
    for name, _ in model.named_parameters():
        if regex.search(reg_exp, name):
            codeword = regex.sub(r'\.(weight|bias)$', '', name)
            target_modules.add(codeword)
    
    return target_modules

def auto_save_path(args):
    args_list = [f'{k}_{v}' for k, v in vars(args).items() if not k.startswith('_')]
    args_string = '_'.join(args_list)
    return f'saves/{args_string}.ckpt'