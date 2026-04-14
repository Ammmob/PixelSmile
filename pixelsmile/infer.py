import argparse
from pathlib import Path

import torch
from PIL import Image

from diffusers import QwenImageEditPlusPipeline

from linear_conditioning import compute_text_embeddings
from utils.image import resize


DEFAULT_METHOD = "score_one_all"
DEFAULT_SEED = 42
DEFAULT_INF_STEPS = 50
DEFAULT_RESIZE_MODE = "crop"
DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 512
DEFAULT_DATA_TYPE = "human"
SUPPORTED_EXPRESSIONS = [
    "angry",
    "confused",
    "contempt",
    "confident",
    "disgust",
    "fear",
    "happy",
    "sad",
    "shy",
    "sleepy",
    "surprised",
    "anxious",
]
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"
DEFAULT_MODEL_PATH = DEFAULT_WEIGHTS_DIR / "Qwen-Image-Edit-2511"
DEFAULT_LORA_PATH = DEFAULT_WEIGHTS_DIR / "PixelSmile.safetensors"


def parse_args():
    parser = argparse.ArgumentParser(description="Run PixelSmile inference on a single image.")
    parser.add_argument(
        "--expression",
        required=True,
        choices=SUPPORTED_EXPRESSIONS,
        help="Target expression to edit.",
    )
    parser.add_argument(
        "--scales",
        type=float,
        nargs="+",
        required=True,
        help="A list of expression strengths to generate.",
    )
    parser.add_argument(
        "--data-type",
        choices=["human", "anime"],
        default=DEFAULT_DATA_TYPE,
        help="Input image type. Defaults to human.",
    )
    parser.add_argument("--image-path", required=True, help="Path to the input image.")
    parser.add_argument("--output-dir", required=True, help="Directory to save edited images.")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH), help="Path to the base model.")
    parser.add_argument("--lora-path", default=str(DEFAULT_LORA_PATH), help="Path to LoRA weights.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    return parser.parse_args()


def load_pipe(model_path: str, lora_path: str, device: torch.device):
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    )

    lora_path = Path(lora_path)
    if not lora_path.exists():
        raise FileNotFoundError(f"LoRA path does not exist: {lora_path}")

    print(f"[INFO] Loading LoRA weights from {lora_path}")
    pipe.load_lora_weights(str(lora_path))
    pipe.to(device)
    return pipe


def get_subject_name(data_type: str) -> str:
    if data_type == "human":
        return "person"
    if data_type == "anime":
        return "character"
    raise ValueError(f"Unsupported data_type: {data_type}")


def build_edit_condition(subject: str, expression: str, scale: float):
    return {
        "prompt": f"Edit the {subject} to show a {expression} expression",
        "prompt_neu": f"Edit the {subject} to show a neutral expression",
        "category": expression,
        "scores": {expression: scale},
    }


def load_input_image(image_path: str):
    image = Image.open(image_path).convert("RGB")
    return resize(image, (DEFAULT_WIDTH, DEFAULT_HEIGHT), DEFAULT_RESIZE_MODE)


def run_edit(pipe, image, expression: str, scale: float, subject: str, seed: int):
    edit_condition = build_edit_condition(subject, expression, scale)
    prompt_embeds, prompt_embeds_mask = compute_text_embeddings(
        method=DEFAULT_METHOD,
        pipeline=pipe,
        data=edit_condition,
        image=image,
        max_sequence_length=1024,
    )

    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    with torch.no_grad():
        return pipe(
            image=image,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            num_inference_steps=DEFAULT_INF_STEPS,
            true_cfg_scale=0,
            output_type="pil",
            generator=generator,
        ).images[0]


def main():
    args = parse_args()

    image_path = Path(args.image_path)
    output_dir = Path(args.output_dir)
    subject = get_subject_name(args.data_type)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pipe = load_pipe(
        model_path=args.model_path,
        lora_path=args.lora_path,
        device=device,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    input_image = load_input_image(str(image_path))

    print(f"[INFO] Input image: {image_path}")
    print(f"[INFO] Model path: {args.model_path}")
    print(f"[INFO] Data type: {args.data_type}")
    print(f"[INFO] Expression: {args.expression}")
    print(f"[INFO] Scales: {args.scales}")
    print(f"[INFO] Seed: {args.seed}")
    print(f"[INFO] Output dir: {output_dir}")

    for scale in args.scales:
        edited_image = run_edit(
            pipe=pipe,
            image=input_image,
            expression=args.expression,
            scale=float(scale),
            subject=subject,
            seed=args.seed,
        )
        save_path = output_dir / f"{image_path.stem}_{args.expression}_{scale}.jpg"
        edited_image.save(save_path)
        print(f"[INFO] Saved to {save_path}")


if __name__ == "__main__":
    main()
