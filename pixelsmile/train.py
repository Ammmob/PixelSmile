import os
import gc
import math
import copy
import random
import argparse
import shutil
from pathlib import Path
from tqdm.auto import tqdm
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import broadcast_object_list
from accelerate.utils import ProjectConfiguration
from accelerate.utils import DistributedDataParallelKwargs

from transformers import (
    CLIPVisionModel, 
    CLIPVisionModelWithProjection, 
    CLIPImageProcessor
)

from diffusers import (
    FlowMatchEulerDiscreteScheduler,
    AutoencoderKLQwenImage,
    QwenImageTransformer2DModel,
    QwenImageEditPlusPipeline,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from diffusers.utils import convert_state_dict_to_diffusers
from diffusers.loaders import AttnProcsLayers

from peft import LoraConfig
from peft.utils import get_peft_model_state_dict

from datasets import load_dataset
from omegaconf import OmegaConf
import bitsandbytes as bnb

from data.dataset import TripletExpressionDataset, collate_fn
from losses.identity import IDLoss
from losses.contrast import SymmetricContrastLoss
from precompute import pre_compute_embeddings
from utils.lora_utils import lora_processors

# Note: If you get a warning about tensorboard, install it with:
# pip install tensorboard

logger = get_logger(__name__, log_level="INFO")

def set_global_seed(seed: int):
    """
    Set random seed for python / numpy / torch (CPU & CUDA).
    Each process uses seed + RANK to avoid identical randomness across GPUs.
    """
    rank = int(os.environ.get("RANK", 0))
    seed = seed + rank

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = True
    
    return seed

def clean_cache(cache_dir):
    print(f"Clean cache dir {cache_dir}...")
    shutil.rmtree(cache_dir)
    print("Clean cache finished.")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="pixelsmile/configs/config.yaml", help="Path to config file")
    return parser.parse_args()

def build_clip_image_encoder(
    model_name="openai/clip-vit-large-patch14",
    dtype=torch.float16,
    device=None,
):
    """
    Build frozen CLIP image encoder (no text encoder)

    Returns
    -------
    vision_model : torch.nn.Module
    processor : CLIPImageProcessor
    """

    vision_model = CLIPVisionModel.from_pretrained(
        model_name,
        torch_dtype=dtype,
    )

    processor = CLIPImageProcessor.from_pretrained(model_name)

    # Keep CLIP encoder frozen.
    vision_model.requires_grad_(False)
    vision_model.eval()

    if device is not None:
        vision_model.to(device)

    return vision_model, processor


_CLIP_MEAN = torch.tensor(
    [0.48145466, 0.4578275, 0.40821073]
).view(1,3,1,1)

_CLIP_STD = torch.tensor(
    [0.26862954, 0.26130258, 0.27577711]
).view(1,3,1,1)


def extract_clip_image_features(
    images,           # [B,C,H,W]  range [-1,1]
    vision_model,
):
    """Extract normalized CLIP image features from [-1, 1] tensors."""

    # CLIP vision expects float32 normalized image inputs.
    images = images.float()

    images = (images + 1) * 0.5
    images = images.clamp(0, 1)

    images = F.interpolate(
        images,
        size=224,
        mode="bilinear",
        align_corners=False,
    )

    mean = _CLIP_MEAN.to(images.device, images.dtype)
    std  = _CLIP_STD.to(images.device, images.dtype)

    images = (images - mean) / std

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        outputs = vision_model(pixel_values=images)


    feats = outputs.pooler_output   # [B, D]

    feats = F.normalize(feats, dim=-1)

    return feats

def calc_id_loss(id_loss_fn, pairs):

    if not isinstance(pairs, (list, tuple)):
        pairs = [pairs]

    losses = []

    for a, b in pairs:

        a_proc = (a.clamp(-1, 1) + 1.0) * 127.5
        b_proc = (b.clamp(-1, 1) + 1.0) * 127.5

        loss = id_loss_fn.compute_id_loss_two_images(a_proc, b_proc)

        losses.append(loss)

    return torch.mean(torch.stack(losses))

def calc_target_loss(
    model_pred_0: torch.Tensor, 
    target_latents_0: torch.Tensor, 
    model_pred_1: torch.Tensor, 
    target_latents_1: torch.Tensor, 
    noise: torch.Tensor, 
    weighting: torch.Tensor
) -> torch.Tensor:
    """
    Compute fully symmetric base reconstruction loss (Symmetric Base Reconstruction Loss).
    """
    def compute_single_branch(model_pred, target_latents):
        target = noise - target_latents
        target = target.permute(0, 2, 1, 3, 4)
        
        loss = torch.mean(
            (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape, -1),
            dim=1,
        ).mean()
        return loss

    loss_0 = compute_single_branch(model_pred_0, target_latents_0)
    loss_1 = compute_single_branch(model_pred_1, target_latents_1)

    return (loss_0 + loss_1) / 2.0

def main(args):
    # Load config
    config = OmegaConf.load(args.config)
    seed = set_global_seed(config.experiment.seed)
    
    # Init accelerator
    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=True, 
        broadcast_buffers=False
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        # log_with=config.logging.report_to,
        kwargs_handlers=[ddp_kwargs]
    )

    container = [None]
    if accelerator.is_main_process:
        container[0] = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    broadcast_object_list(container, from_process=0)
    timestamp = container[0]

    
    exp_root = os.path.join("exps", timestamp)

    exp_root = exp_root
    ckpt_dir = os.path.join(exp_root, "ckpts")
    logging_dir = os.path.join(exp_root, "logs")
    cache_dir = os.path.join(exp_root, "cache")
    config_dir = os.path.join(exp_root, "configs")

    if accelerator.is_main_process:
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(logging_dir, exist_ok=True)
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(config_dir, exist_ok=True)
        snapshot_path = os.path.join(
            config_dir,
            "config.yaml"
        )
        OmegaConf.save(config, snapshot_path)
        print("\n================ Experiment =================")
        print(f"Root: {exp_root}")
        print("============================================\n")

    
    accelerator_project_config = ProjectConfiguration(
        project_dir=ckpt_dir, 
        logging_dir=logging_dir
    )
    accelerator.project_configuration = accelerator_project_config
    accelerator.log_with = [config.logging.report_to]
    accelerator.init_trackers(config.logging.tracker_project_name)
    
    # Setup logging
    logger.info(accelerator.state, main_process_only=False)
    
        
    # Setup weight dtype
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        
    # Load clip
    vision_model, processor = build_clip_image_encoder(
        model_name=config.model.clip_path,
        dtype=torch.float16,
        device=accelerator.device,
    )

    # vision_model = accelerator.prepare(vision_model)
        
    # Load models for embedding computation
    if accelerator.is_main_process:
        print("Loading models for embedding computation...")
    text_encoding_pipeline = QwenImageEditPlusPipeline.from_pretrained(
        config.model.pretrained_path,
        transformer=None,
        vae=None,
        torch_dtype=weight_dtype
    )
    text_encoding_pipeline.to(accelerator.device)
    
    vae = AutoencoderKLQwenImage.from_pretrained(
        config.model.pretrained_path,
        subfolder="vae",
    )
    vae.to(accelerator.device, dtype=weight_dtype)
    
    # Load dataset
    if accelerator.is_main_process:
        print("Loading dataset...")
    raw_dataset = load_dataset(
        "json",
        data_files=config.dataset.path,
        split="train"
    )
    
    if accelerator.is_main_process:
        print(f"Pre-computing embeddings in cache dir {cache_dir}...")
        
    pre_compute_embeddings(
        raw_dataset,
        text_encoding_pipeline,
        vae,
        accelerator,
        save_dir=cache_dir,
        data_type=config.dataset.data_type,
        method=config.experiment.method,
        max_samples=config.dataset.max_samples,
        width=config.dataset.resolution.width,
        height=config.dataset.resolution.height,
        resize_mode=config.dataset.resize_mode,
        use_score_rescale=config.dataset.use_score_rescale
    )
    
    # Clean up encoding models
    del text_encoding_pipeline
    del vae
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"Process {accelerator.process_index} pre-computing finished, waiting for others...")
    accelerator.wait_for_everyone() 
    vae = AutoencoderKLQwenImage.from_pretrained(
        config.model.pretrained_path,
        subfolder="vae"
    )
    vae.to(device=accelerator.device, dtype=weight_dtype)
    # accelerator.state.select_deepspeed_plugin("vae")
    vae.requires_grad_(False)
    vae.eval()
    # VAE needs to be prepared to handle device placement/dtype in DS Zero-3 env
    vae = accelerator.prepare(vae)
    
    # Load transformer for training
    if accelerator.is_main_process:
        print("Loading transformer model...")
    transformer = QwenImageTransformer2DModel.from_pretrained(
        config.model.pretrained_path,
        subfolder="transformer",
        torch_dtype=weight_dtype,
        low_cpu_mem_usage=True
    )

    # Setup LoRA
    lora_config = LoraConfig(
        r=config.lora.rank,
        lora_alpha=config.lora.alpha,
        init_lora_weights="gaussian",
        target_modules=config.lora.target_modules,
        lora_dropout=config.lora.dropout,
    )
    transformer.add_adapter(lora_config)
    
    # Setup noise scheduler
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        config.model.pretrained_path,
        subfolder="scheduler",
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    
    # Prepare for training
    transformer.requires_grad_(False)
    transformer.train()
    
    # Only train LoRA parameters
    for n, param in transformer.named_parameters():
        if 'lora' in n:
            param.requires_grad = True
            # print(f"Training: {n}")
        else:
            param.requires_grad = False
            
    trainable_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    if accelerator.is_main_process:
        print(f"Trainable parameters: {trainable_params / 1_000_000:.2f}M")
    
    # Enable gradient checkpointing
    transformer.enable_gradient_checkpointing()
    
    # Setup optimizer
    lora_layers = filter(lambda p: p.requires_grad, transformer.parameters())
    
    if config.training.optimizer.use_8bit_adam:
        optimizer = bnb.optim.Adam8bit(
            lora_layers,
            lr=config.training.learning_rate,
            betas=(config.training.optimizer.beta1, config.training.optimizer.beta2),
        )
    else:
        optimizer = torch.optim.AdamW(
            lora_layers,
            lr=config.training.learning_rate,
            betas=(config.training.optimizer.beta1, config.training.optimizer.beta2),
            weight_decay=config.training.optimizer.weight_decay,
            eps=config.training.optimizer.epsilon,
        )
    
    # Create dataset and dataloader
    train_dataset = TripletExpressionDataset(
        dataset_path=config.dataset.path,
        cache_dir=cache_dir
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.dataset.num_workers,
        generator=torch.Generator().manual_seed(seed),
    )
    
    # Prepare for accelerator
    lora_layers_model = AttnProcsLayers(lora_processors(transformer))
    transformer, optimizer, train_dataloader = accelerator.prepare(
        transformer,
        optimizer,
        train_dataloader,
    )
    
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config.training.gradient_accumulation_steps)
    max_train_steps = config.training.num_epochs * num_update_steps_per_epoch
  
    # Setup learning rate scheduler
    lr_scheduler = get_scheduler(
        config.training.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.training.lr_warmup_steps,
        num_training_steps=max_train_steps,
    )
    lr_scheduler = accelerator.prepare(lr_scheduler)
    
    # Load VAE config for scaling
    vae_config = AutoencoderKLQwenImage.load_config(
        config.model.pretrained_path,
        subfolder="vae"
    )
    # Check for the correct key - it might be 'temporal_downsample' or 'temperal_downsample' or in a different structure
    if 'temporal_downsample' in vae_config:
        vae_scale_factor = 2 ** len(vae_config['temporal_downsample'])
    elif 'temperal_downsample' in vae_config:  # Check for typo version
        vae_scale_factor = 2 ** len(vae_config['temperal_downsample'])
    else:
        # Default value based on typical Qwen VAE architecture
        logger.warning("Could not find temporal_downsample in VAE config, using default scale factor of 8")
        vae_scale_factor = 8
    
    # Training loop
    global_step = 0
    
    # Initialize wandb if requested
    if accelerator.is_main_process and config.logging.report_to == "wandb":
        import wandb
        wandb.init(
            project=config.logging.tracker_project_name,
            name=timestamp,
            config={
                "learning_rate": config.training.learning_rate,
                "epochs": config.training.num_epochs,
                "batch_size": config.training.batch_size,
                "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
                "lora_rank": config.lora.rank,
                "lora_alpha": config.lora.alpha,
                "dataset_size": len(train_dataset),
            }
        )
    
    
    progress_bar = tqdm(
        range(max_train_steps),
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    
    
    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma
    
    # Preapare Loss
    contrast_loss_weight = config.training.get("contrast_loss_weight", 0.0)
    id_loss_weight = config.training.get("id_loss_weight", 0.0)
    if config.dataset.data_type != "human":
        id_loss_weight = 0.0
    use_contrast_loss = contrast_loss_weight > 0.0
    use_id_loss = id_loss_weight > 0.0
    
    if use_contrast_loss:
        contrast_loss_fn = SymmetricContrastLoss(
            mode=config.training.contrast_loss_mode,
            symmetric=config.training.contrast_loss_symmetric,
        )
    
    if use_id_loss:
        # Select plugin for ID Loss components (usually frozen backbone)
        # accelerator.state.select_deepspeed_plugin("id_loss")
        id_loss_fn = IDLoss(
            device=accelerator.device,
            det_model_path=config.model.insightface_detector_path,
            rec_model_path=config.model.insightface_recognition_path,
        )
        # Prepare ID Loss network
        id_loss_fn.netArc = accelerator.prepare(id_loss_fn.netArc)
        
    if accelerator.is_main_process:
        print("\n================ Config =================")
        print("Training desc:", config.experiment.description)
        print("Training method:", config.experiment.method)
        print("Training loss:")
        print("Contrast Loss mode", config.training.contrast_loss_mode)
        print("Contrast Loss Weight:", contrast_loss_weight)
        print("ID Loss Weight:", id_loss_weight)
        print("============================================\n")
    
    for epoch in range(config.training.num_epochs):
        train_loss = 0.0
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(transformer):
                # Get batch data
                target_latents_0 = batch['target_latents_0'].to(dtype=weight_dtype, device=accelerator.device)
                target_latents_1 = batch['target_latents_1'].to(dtype=weight_dtype, device=accelerator.device)
                control_latents = batch['control_latents'].to(dtype=weight_dtype, device=accelerator.device)
                prompt_embeds_0 = batch['prompt_embeds_0'].to(dtype=weight_dtype, device=accelerator.device)
                prompt_embeds_mask_0 = batch['prompt_embeds_mask_0'].to(dtype=torch.int32, device=accelerator.device)
                prompt_embeds_1 = batch['prompt_embeds_1'].to(dtype=weight_dtype, device=accelerator.device)
                prompt_embeds_mask_1 = batch['prompt_embeds_mask_1'].to(dtype=torch.int32, device=accelerator.device)
                
                # Prepare latents
                target_latents_0 = target_latents_0.permute(0, 2, 1, 3, 4)
                target_latents_1 = target_latents_1.permute(0, 2, 1, 3, 4)
                control_latents = control_latents.permute(0, 2, 1, 3, 4)
                
                # Normalize latents
                if 'latents_mean' in vae_config and 'latents_std' in vae_config and 'z_dim' in vae_config:
                    latents_mean = (
                        torch.tensor(vae_config['latents_mean'])
                        .view(1, 1, vae_config['z_dim'], 1, 1)
                        .to(control_latents.device, control_latents.dtype)
                    )
                    latents_std = 1.0 / torch.tensor(vae_config['latents_std']).view(
                        1, 1, vae_config['z_dim'], 1, 1
                    ).to(control_latents.device, control_latents.dtype)
                    
                    target_latents_0 = (target_latents_0 - latents_mean) * latents_std
                    target_latents_1 = (target_latents_1 - latents_mean) * latents_std
                    control_latents = (control_latents - latents_mean) * latents_std
                else:
                    # Skip normalization if config doesn't have the required fields
                    logger.warning("VAE config missing latents normalization parameters, skipping normalization")
                
                # Add noise
                bsz = control_latents.shape[0]
                noise = torch.randn_like(control_latents)
                
                # Sample timesteps
                u = compute_density_for_timestep_sampling(
                    weighting_scheme="none",
                    batch_size=bsz,
                    logit_mean=0.0,
                    logit_std=1.0,
                    mode_scale=1.29,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=control_latents.device)
                
                # Get sigmas
                sigmas = get_sigmas(timesteps, n_dim=control_latents.ndim, dtype=control_latents.dtype)
      
                noisy_model_input_0 = (1.0 - sigmas) * target_latents_0 + sigmas * noise
                noisy_model_input_1 = (1.0 - sigmas) * target_latents_1 + sigmas * noise
                
                # Pack latents
                packed_noisy_input_0 = QwenImageEditPlusPipeline._pack_latents(
                    noisy_model_input_0,
                    bsz,
                    noisy_model_input_0.shape[2],
                    noisy_model_input_0.shape[3],
                    noisy_model_input_0.shape[4],
                )
                packed_noisy_input_1 = QwenImageEditPlusPipeline._pack_latents(
                    noisy_model_input_1,
                    bsz,
                    noisy_model_input_1.shape[2],
                    noisy_model_input_1.shape[3],
                    noisy_model_input_1.shape[4],
                )
                packed_control = QwenImageEditPlusPipeline._pack_latents(
                    control_latents,
                    bsz,
                    control_latents.shape[2],
                    control_latents.shape[3],
                    control_latents.shape[4],
                )
                
                # Concatenate for image editing
                packed_input_concat_0 = torch.cat([packed_noisy_input_0, packed_control], dim=1)
                packed_input_concat_1 = torch.cat([packed_noisy_input_1, packed_control], dim=1)
                
                # Image shapes for RoPE
                img_shapes = [[(1, noisy_model_input_0.shape[3] // 2, noisy_model_input_0.shape[4] // 2),
                              (1, control_latents.shape[3] // 2, control_latents.shape[4] // 2)]] * bsz
                
                # Text sequence lengths
                txt_seq_lens_0 = prompt_embeds_mask_0.sum(dim=1).tolist()
                txt_seq_lens_1 = prompt_embeds_mask_1.sum(dim=1).tolist()
                
                # Forward pass
                model_pred_0 = transformer(
                    hidden_states=packed_input_concat_0,
                    timestep=timesteps / 1000,
                    guidance=None,
                    encoder_hidden_states_mask=prompt_embeds_mask_0,
                    encoder_hidden_states=prompt_embeds_0,
                    img_shapes=img_shapes,
                    txt_seq_lens=txt_seq_lens_0,
                    return_dict=False,
                )[0]
                model_pred_1 = transformer(
                    hidden_states=packed_input_concat_1,
                    timestep=timesteps / 1000,
                    guidance=None,
                    encoder_hidden_states_mask=prompt_embeds_mask_1,
                    encoder_hidden_states=prompt_embeds_1,
                    img_shapes=img_shapes,
                    txt_seq_lens=txt_seq_lens_1,
                    return_dict=False,
                )[0]
                
                model_pred_0 = model_pred_0.to(noisy_model_input_0.dtype)
                model_pred_1 = model_pred_1.to(noisy_model_input_1.dtype)
                
                # Extract prediction for target
                model_pred_0 = model_pred_0[:, :packed_noisy_input_0.size(1)]
                model_pred_1 = model_pred_1[:, :packed_noisy_input_1.size(1)]
                
                # Unpack
                model_pred_0 = QwenImageEditPlusPipeline._unpack_latents(
                    model_pred_0,
                    height=noisy_model_input_0.shape[3] * vae_scale_factor,
                    width=noisy_model_input_0.shape[4] * vae_scale_factor,
                    vae_scale_factor=vae_scale_factor,
                )
                model_pred_1 = QwenImageEditPlusPipeline._unpack_latents(
                    model_pred_1,
                    height=noisy_model_input_1.shape[3] * vae_scale_factor,
                    width=noisy_model_input_1.shape[4] * vae_scale_factor,
                    vae_scale_factor=vae_scale_factor,
                )
                
                weighting = compute_loss_weighting_for_sd3(weighting_scheme="none", sigmas=sigmas)
                
                loss = calc_target_loss(
                    model_pred_0=model_pred_0,
                    target_latents_0=target_latents_0,
                    model_pred_1=model_pred_1,
                    target_latents_1=target_latents_1,
                    noise=noise,
                    weighting=weighting
                )
                
                # get x_0 
                curr_noisy_latents_0 = noisy_model_input_0.permute(0, 2, 1, 3, 4)
                curr_noisy_latents_1 = noisy_model_input_1.permute(0, 2, 1, 3, 4)
                gt_latents_0 = target_latents_0.permute(0, 2, 1, 3, 4)
                gt_latents_1 = target_latents_1.permute(0, 2, 1, 3, 4)
                sigmas_broad = sigmas.view(curr_noisy_latents_0.shape[0], 1, 1, 1, 1)
                x_pred_latents_0 = curr_noisy_latents_0 - sigmas_broad * model_pred_0
                x_pred_latents_1 = curr_noisy_latents_1 - sigmas_broad * model_pred_1
            

                latents_mean_unpermuted = latents_mean.permute(0, 2, 1, 3, 4)
                latents_std_unpermuted = latents_std.permute(0, 2, 1, 3, 4)                
                
                
                def vae_decode(latents):
                    latents = latents / latents_std_unpermuted + latents_mean_unpermuted 
                    return vae.decode(latents, return_dict=False)[0][:, :, 0]
                    

                gt_pixels_0 = vae_decode(gt_latents_0)
                gt_pixels_1 = vae_decode(gt_latents_1)
                pred_pixels_0 = vae_decode(x_pred_latents_0)
                pred_pixels_1 = vae_decode(x_pred_latents_1)
                
                with torch.no_grad():
                    gt_feature_0 = extract_clip_image_features(
                        images=gt_pixels_0,
                        vision_model=vision_model,               
                    )
                    gt_feature_1 = extract_clip_image_features(
                        images=gt_pixels_1,
                        vision_model=vision_model,                    
                    )
                pred_feature_0 = extract_clip_image_features(
                    images=pred_pixels_0,
                    vision_model=vision_model,
                )
                pred_feature_1 = extract_clip_image_features(
                    images=pred_pixels_1,
                    vision_model=vision_model,
                )

                if use_contrast_loss:
                    contrast_loss_val = contrast_loss_fn(
                        pred_feature_0,
                        gt_feature_0,
                        pred_feature_1,
                        gt_feature_1,
                    )
                    loss = loss + contrast_loss_weight * contrast_loss_val
                
                if use_id_loss:
                    id_loss_val = calc_id_loss(
                        id_loss_fn,
                        pairs = [
                            (gt_pixels_0, pred_pixels_0),
                            (gt_pixels_1, pred_pixels_1)
                        ]
                    )

                    current_t = timesteps[0].item()
                    
                    if current_t > 600: 
                        epsilon = 1e-5
                    else:
                        epsilon = 1.0 - 1e-5 # Adjusted from 1 - 1e-5 to float
                    id_loss_val = epsilon * id_loss_val
                        
                    loss = loss + id_loss_weight * id_loss_val 
                    
                # Gather loss
                avg_loss = accelerator.gather(loss.detach()).mean()

                train_loss += avg_loss.item() / config.training.gradient_accumulation_steps
                
                # Backward
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(transformer.parameters(), config.training.max_grad_norm)
                    
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
            
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                # Log to wandb
                if accelerator.is_main_process:
                    log_dict = {
                        "train_loss": train_loss,
                        "learning_rate": lr_scheduler.get_last_lr()[0],
                        "global_step": global_step,
                        "grad_norm": grad_norm.item(),
                    }
                    
                    if config.logging.report_to == "wandb":
                        import wandb
                        wandb.log(log_dict)
                    
                    accelerator.log(log_dict, step=global_step)
                
                    train_loss = 0.0
                    
                # Save checkpoint
                if global_step % (config.logging.checkpointing_steps) == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(ckpt_dir, f"checkpoint-{global_step}")
                        os.makedirs(save_path, exist_ok=True)
        
                        # Save LoRA weights
                        unwrapped_transformer = accelerator.unwrap_model(transformer)
            
                        transformer_lora_state_dict = convert_state_dict_to_diffusers(
                            get_peft_model_state_dict(unwrapped_transformer)
                        )
                        
                        QwenImageEditPlusPipeline.save_lora_weights(
                            save_path,
                            transformer_lora_state_dict,
                            safe_serialization=True,
                        )
                        
                        logger.info(f"Saved checkpoint to {save_path}")
                    
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
        
    
    # Save final model
    if accelerator.is_main_process:
        save_path = os.path.join(ckpt_dir, "final_lora")
        os.makedirs(save_path, exist_ok=True)
        
        unwrapped_transformer = accelerator.unwrap_model(transformer)
        transformer_lora_state_dict = convert_state_dict_to_diffusers(
            get_peft_model_state_dict(unwrapped_transformer)
        )
        
        QwenImageEditPlusPipeline.save_lora_weights(
            save_path,
            transformer_lora_state_dict,
            safe_serialization=True,
        )
        
        logger.info(f"Saved final model to {save_path}")
        
        clean_cache(cache_dir)
    
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == '__main__':
    args = parse_args()
    main(args)
