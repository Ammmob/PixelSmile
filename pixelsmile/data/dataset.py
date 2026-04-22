import os
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from typing import Optional


class TripletExpressionDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        cache_dir: str,
        max_samples: Optional[int] = None,
    ):
        self.cache_dir = cache_dir

        self.dataset = load_dataset(
            "json",
            data_files=dataset_path,
            split="train"
        )

        if max_samples:
            self.dataset = self.dataset.select(
                range(min(max_samples, len(self.dataset)))
            )

        print(f"Dataset size: {len(self.dataset)}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        key = f"sample_{idx}"
        path = os.path.join(self.cache_dir, f"{key}.pt")
        embeds = torch.load(path, map_location="cpu", weights_only=True)

        return {
            "target_latents_0": embeds["target_latents_0"],
            "target_latents_1": embeds["target_latents_1"],
            "control_latents": embeds["control_latents"],
            "prompt_embeds_0": embeds["prompt_embeds_0"],
            "prompt_embeds_mask_0": embeds["prompt_embeds_mask_0"],
            "prompt_embeds_1": embeds["prompt_embeds_1"],
            "prompt_embeds_mask_1": embeds["prompt_embeds_mask_1"],
        }


# ==========================================================
# Collate
# ==========================================================

def _pad_prompt_batch(emb_list, mask_list):
    max_len = max(e.shape[0] for e in emb_list)

    padded_emb = []
    padded_mask = []

    for emb, mask in zip(emb_list, mask_list):
        seq = emb.shape[0]

        if seq < max_len:
            pad_e = torch.zeros(
                max_len - seq,
                emb.shape[1],
                dtype=emb.dtype
            )
            emb = torch.cat([emb, pad_e], dim=0)

            pad_m = torch.zeros(
                max_len - seq,
                dtype=mask.dtype
            )
            mask = torch.cat([mask, pad_m], dim=0)

        padded_emb.append(emb)
        padded_mask.append(mask)

    return torch.stack(padded_emb), torch.stack(padded_mask)


def collate_fn(examples):

    # ===== Latents =====
    batch = {
        "target_latents_0": torch.stack([e["target_latents_0"] for e in examples]),
        "target_latents_1": torch.stack([e["target_latents_1"] for e in examples]),
        "control_latents": torch.stack([e["control_latents"] for e in examples]),
    }

    # ===== Prompt Branch 0 =====
    emb0, mask0 = _pad_prompt_batch(
        [e["prompt_embeds_0"] for e in examples],
        [e["prompt_embeds_mask_0"] for e in examples],
    )

    # ===== Prompt Branch 1 =====
    emb1, mask1 = _pad_prompt_batch(
        [e["prompt_embeds_1"] for e in examples],
        [e["prompt_embeds_mask_1"] for e in examples],
    )

    batch.update({
        "prompt_embeds_0": emb0,
        "prompt_embeds_mask_0": mask0,
        "prompt_embeds_1": emb1,
        "prompt_embeds_mask_1": mask1,
    })

    return batch
