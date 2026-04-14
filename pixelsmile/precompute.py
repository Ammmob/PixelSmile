import os
import torch
import torch.nn.functional as F
import numpy as np
import gc
from PIL import Image
from tqdm.auto import tqdm
from typing import Optional

from linear_conditioning import compute_text_embeddings
from utils.image import resize, scale_scores
    

def pre_compute_embeddings(
    dataset,
    pipeline,
    vae,
    accelerator,
    save_dir: str, 
    data_type: str,
    method: str = "direct",
    max_samples: Optional[int] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    resize_mode: str = "padding",
    use_score_rescale: bool = False,
):
    """
    预计算 Text Embedding 和 Image Latents 并保存到磁盘。
    支持断点续传和多卡并行计算。
    """

    num_samples = min(len(dataset), max_samples) if max_samples else len(dataset)
    
    # 模拟输入以初始化模型组件，避免首次推理卡顿或报错
    dummy_input = torch.zeros(1, 3, 1, 64, 64).to(device=accelerator.device, dtype=vae.dtype)
    _ = vae.encode(dummy_input)
    
    # 任务分片逻辑
    compute_list = []
    for idx in range(num_samples):
        # 多卡环境下，只处理分配给当前进程的索引
        if idx % accelerator.num_processes != accelerator.process_index:
            continue
        
        key = f"sample_{idx}"
        embeds_path = os.path.join(save_dir, f"{key}.pt")

        # 断点续传：如果文件已存在，跳过
        if os.path.exists(embeds_path):
            continue
        compute_list.append((idx, embeds_path))
        
    accelerator.wait_for_everyone()
    total = accelerator.gather_for_metrics(
        torch.tensor(len(compute_list), device=accelerator.device)
    ).sum()

    if accelerator.is_main_process:
        print("Remaining tasks:", total.item())
        
    with torch.no_grad():
        # tqdm 进度条只在每个进程本地显示
        for idx, embeds_path in tqdm(compute_list, desc=f"GPU {accelerator.process_index} Pre-computing", disable=not accelerator.is_local_main_process):

            item = dataset[idx]
            
            # 缩放表情分数
            if use_score_rescale:
                item = scale_scores(item)
            
            control_image = Image.open(item['src']['image']).convert('RGB')
            target_image_0 = Image.open(item['dst'][0]['image']).convert('RGB')
            target_image_1 = Image.open(item['dst'][1]['image']).convert('RGB')
            
    
            control_image = resize(control_image, (width, height), resize_mode, box=item['src']['boxes'])
            target_image_0 = resize(target_image_0, (width, height), resize_mode, box=item['dst'][0]['boxes'])
            target_image_1 = resize(target_image_1, (width, height), resize_mode, box=item['dst'][1]['boxes'])

            if data_type == "human":
                subject = "person"
            elif data_type == "anime":
                subject = "character"
            else:
                raise ValueError(f"data_type {data_type} not exists")
            
            prompt_0 = f"Edit the {subject} to show a {item['dst'][0]['category']} expression"
            prompt_1 = f"Edit the {subject} to show a {item['dst'][1]['category']} expression"
            prompt_neu = f"Edit the {subject} to show a neutral expression"

            
            prompt_embeds_0, prompt_embeds_mask_0 = compute_text_embeddings(
                method=method,
                pipeline=pipeline,
                data = {
                    "category" : item['dst'][0]['category'],
                    "scores": item["dst"][0]["scores"],
                    "prompt": prompt_0,
                    "prompt_neu": prompt_neu
                },
                image=control_image,
                max_sequence_length=1024
            )
            prompt_embeds_1, prompt_embeds_mask_1 = compute_text_embeddings(
                method=method,
                pipeline=pipeline,
                data = {
                    "category" : item['dst'][1]['category'],
                    "scores": item["dst"][1]["scores"],
                    "prompt": prompt_1,
                    "prompt_neu": prompt_neu
                },
                image=control_image,
                max_sequence_length=1024
            )

            # 2. 编码图像 (VAE) -> Latents
            def encode_vae(img):
                img_tensor = (torch.from_numpy(np.array(img).astype(np.float32)) / 127.5) - 1.0
                pixel_values = img_tensor.permute(2, 0, 1).unsqueeze(0).unsqueeze(2) # [1, 3, 1, H, W]
                return vae.encode(pixel_values.to(dtype=vae.dtype, device=accelerator.device)).latent_dist.sample()[0].cpu()

            # 原子化保存：先写临时文件再重命名，防止中断导致坏文件
            tmp_path = embeds_path + ".tmp"
            torch.save({
                'prompt_embeds_0': prompt_embeds_0[0].cpu(),
                'prompt_embeds_mask_0': prompt_embeds_mask_0[0].cpu(),
                'prompt_embeds_1': prompt_embeds_1[0].cpu(),
                'prompt_embeds_mask_1': prompt_embeds_mask_1[0].cpu(),
                'target_latents_0' : encode_vae(target_image_0),
                'target_latents_1' : encode_vae(target_image_1),
                'control_latents' : encode_vae(control_image)
            }, tmp_path)
            os.replace(tmp_path, embeds_path)
            
            # 显存清理
            if idx % 50 == 0:
                torch.cuda.empty_cache()
                gc.collect()
    
    # 确保所有计算完毕           
    accelerator.wait_for_everyone()
