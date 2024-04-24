import torch
import torch.nn.functional as F
from transformers import SamConfig, SamModel

def custom_model(pretrained_path, image_size, num_classes, vit_patch_size):
    config = SamConfig.from_pretrained(pretrained_path)

    config.vision_config.patch_size = vit_patch_size
    config.vision_config.image_size = image_size

    config.mask_decoder_config.num_multimask_outputs = num_classes

    config.prompt_encoder_config.patch_size = vit_patch_size
    config.prompt_encoder_config.image_size = image_size
    return SamModel(config)

def pretrain_state_dict(pretrained_path):
    model = SamModel.from_pretrained(pretrained_path)
    return model.state_dict()

def resize_patch_embed(current_patch_embed, ckpt_patch_embed):
    reshape_size = (1, current_patch_embed.shape[-2], current_patch_embed.shape[-1])
    ckpt_patch_embed = torch.unsqueeze(ckpt_patch_embed, dim=2)
    ckpt_patch_embed = F.interpolate(ckpt_patch_embed, size=reshape_size, mode='trilinear', align_corners=False)
    return torch.squeeze(ckpt_patch_embed, dim=2)

def resize_pos_embed(pos_embed, token_size):
    pos_embed = torch.permute(pos_embed, (0, 3, 1, 2))
    pos_embed = F.interpolate(pos_embed, (token_size, token_size), mode='bilinear', align_corners=False)
    return torch.permute(pos_embed, (0, 2, 3, 1))

def resize_rel_pos(rel_pos_params, token_size):
    _, w = rel_pos_params.shape
    rel_pos_params = torch.unsqueeze(rel_pos_params, dim=0)
    rel_pos_params = torch.unsqueeze(rel_pos_params, dim=0)
    
    rel_pos_params = F.interpolate(rel_pos_params, (token_size * 2 - 1, w), mode='bilinear', align_corners=False)
    return rel_pos_params[0, 0]

def load_from(model, pretrain_state, image_size, vit_patch_size):
    current_state = model.state_dict()

    except_keys = ['mask_tokens', 'output_hypernetworks_mlps', 'iou_prediction_head']
    ckpt_state = {k: v for k, v in pretrain_state.items() if not except_keys[0] in k and not except_keys[1] in k and not except_keys[2] in k}

    patch_key = 'vision_encoder.patch_embed.projection.weight'
    if ckpt_state[patch_key].shape[-2:] != current_state[patch_key].shape[-2:]:
        ckpt_state[patch_key] = resize_patch_embed(current_state[patch_key], ckpt_state[patch_key])

    pos_key = 'vision_encoder.pos_embed'
    token_size = int(image_size // vit_patch_size)

    if ckpt_state[pos_key].shape[1] != token_size:
        ckpt_state[pos_key] = resize_pos_embed(ckpt_state[pos_key], token_size)

        rel_pos_keys = [k for k in current_state.keys() if 'rel_pos' in k]
        global_rel_pos_keys = [k for k in rel_pos_keys if '2' in k or '5' in  k or '8' in k or '11' in k]

        for k in global_rel_pos_keys:
            ckpt_state[k] = resize_rel_pos(ckpt_state[k], token_size)

    model.load_state_dict(ckpt_state, strict=False)
    return model

def build_sam(pretrained_path, num_classes, image_size=1024, vit_patch_size=16):
    model = custom_model(pretrained_path, image_size, num_classes, vit_patch_size)
    state = pretrain_state_dict(pretrained_path)
    return load_from(model, state, image_size, vit_patch_size)