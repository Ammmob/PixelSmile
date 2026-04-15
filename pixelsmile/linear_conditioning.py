import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

def compute_text_embeddings(
    method: str,
    pipeline: QwenImageEditPlusPipeline,
    data: dict,
    image: Image,
    max_sequence_length:int = 1024
):
    valid_methods = ["direct", "score_scale", "score_one", "score_one_exp", "score_one_tgt", "score_one_all", "score_multi_all"]
    if method == "direct":
        return _compute_direct(pipeline, data, image, max_sequence_length)
    elif method == "score_scale":
        return _compute_score_scale(pipeline, data, image, max_sequence_length)
    elif method == "score_one":
        return _compute_score_one(pipeline, data, image, max_sequence_length)
    elif method == "score_one_exp":
        return _compute_score_one_exp(pipeline, data, image, max_sequence_length)
    elif method == "score_one_tgt":
        return _compute_score_one_tgt(pipeline, data, image, max_sequence_length)
    elif method == "score_one_all":
        return _compute_score_one_all(pipeline, data, image, max_sequence_length)
    else:
        raise ValueError(f"Unknow method: {method}, method should in {valid_methods}")

def _compute_direct(
    pipeline: QwenImageEditPlusPipeline,
    data: dict,
    image: Image,
    max_sequence_length:int,
):
    prompt = data["prompt"]
        
    prompt_embeds, prompt_embeds_mask = pipeline.encode_prompt(
        image=image,
        prompt=prompt,
        device=pipeline.device,
        num_images_per_prompt=1,
        max_sequence_length=max_sequence_length,
    )
    return prompt_embeds, prompt_embeds_mask

def _compute_score_scale(
    pipeline: QwenImageEditPlusPipeline,
    data: dict,
    image: Image,
    max_sequence_length: int = 1024
):
    device = pipeline.device
    
    target_cat = data["category"]
    score = data["scores"][target_cat]
    prompt_tgt = data["prompt"]
    
    with torch.no_grad():
        embed_tgt, mask_tgt = pipeline.encode_prompt(
            prompt=prompt_tgt,
            image=image,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=max_sequence_length
        )
    
    embed_tgt[:, -7] *= score

    return embed_tgt, mask_tgt

    
def _compute_score_one(
    pipeline: QwenImageEditPlusPipeline,
    data: dict,
    image: Image,
    max_sequence_length: int = 1024
):
    device = pipeline.device
    
    target_cat = data["category"]
    score = data["scores"][target_cat]
    prompt_tgt = data["prompt"]
    prompt_neu = data["prompt_neu"]
    
    with torch.no_grad():
        embed_neu, mask_neu = pipeline.encode_prompt(
            prompt=prompt_neu,
            image=image, 
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=max_sequence_length
        )
        
        embed_tgt, mask_tgt = pipeline.encode_prompt(
            prompt=prompt_tgt,
            image=image,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=max_sequence_length
        )
    
    prefix = embed_tgt[:, :-7, :]  
    
    suffix_neu = embed_neu[:, -7:, :]
    suffix_tgt = embed_tgt[:, -7:, :]
    
    delta = suffix_tgt - suffix_neu
    suffix = suffix_neu + score * delta
    
    final_embeds = torch.cat([prefix, suffix], dim=1)
    
    final_mask = mask_tgt
    
    return final_embeds, final_mask
    
def _compute_score_one_exp(
    pipeline: QwenImageEditPlusPipeline,
    data: dict,
    image: Image,
    max_sequence_length: int = 1024
):
    device = pipeline.device
    
    target_cat = data["category"]
    score = data["scores"][target_cat]
    prompt_tgt = data["prompt"]
    prompt_neu = data["prompt_neu"]
    
    with torch.no_grad():
        embed_neu, mask_neu = pipeline.encode_prompt(
            prompt=prompt_neu,
            image=image, 
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=max_sequence_length
        )
        
        embed_tgt, mask_tgt = pipeline.encode_prompt(
            prompt=prompt_tgt,
            image=image,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=max_sequence_length
        )
    
    prefix = embed_tgt[:, :-6, :]  
    
    suffix_neu = embed_neu[:, -6:, :]
    suffix_tgt = embed_tgt[:, -6:, :]
    
    delta = suffix_tgt - suffix_neu
    suffix = suffix_neu + score * delta
    
    final_embeds = torch.cat([prefix, suffix], dim=1)
    
    final_mask = mask_tgt
    
    return final_embeds, final_mask

def _compute_score_one_tgt(
    pipeline: QwenImageEditPlusPipeline,
    data: dict,
    image: Image,
    max_sequence_length: int = 1024
):
    device = pipeline.device
    
    target_cat = data["category"]
    score = data["scores"][target_cat]
    prompt_tgt = data["prompt"]
    prompt_neu = data["prompt_neu"]
    
    with torch.no_grad():
        embed_neu, mask_neu = pipeline.encode_prompt(
            prompt=prompt_neu,
            image=image, 
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=max_sequence_length
        )
        
        embed_tgt, mask_tgt = pipeline.encode_prompt(
            prompt=prompt_tgt,
            image=image,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=max_sequence_length
        )
    

    exp_neu = embed_neu[:, -7, :]
    exp_tgt = embed_tgt[:, -7, :]
    
    delta = exp_tgt - exp_neu
    exp = exp_neu + score * delta
    
    embed_tgt[:, -7] = exp

    return embed_tgt, mask_tgt

def _compute_score_one_all(
    pipeline: QwenImageEditPlusPipeline,
    data: dict,
    image: Image,
    max_sequence_length: int = 1024
):
    device = pipeline.device
    
    target_cat = data["category"]
    score = data["scores"][target_cat]
    prompt_tgt = data["prompt"]
    prompt_neu = data["prompt_neu"]
    
    with torch.no_grad():
        embed_neu, mask_neu = pipeline.encode_prompt(
            prompt=prompt_neu,
            image=image, 
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=max_sequence_length
        )
        
        embed_tgt, mask_tgt = pipeline.encode_prompt(
            prompt=prompt_tgt,
            image=image,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=max_sequence_length
        )
    

    
    delta = embed_tgt - embed_neu
    embed_tgt = embed_neu + score * delta
    
    return embed_tgt, mask_tgt
