#!/usr/bin/env python3
"""Run BFS face swap from command line.

Usage:
    python swap_face.py --face ./face.jpg --person ./person.jpg
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any, Callable

import torch
from diffusers import AutoPipelineForImage2Image
from PIL import Image
from safetensors.torch import load_file

BASE_MODEL = "Qwen/Qwen-Image-Edit-2511"
LORA_REPO = "Alissonerdx/BFS-Best-Face-Swap"

# BFS V3/V4/V5 use inverted order (Image 1 = body/person, Image 2 = face)
# LORA_VARIANTS = {
#     "face_v1": {
#         "filename": "bfs_face_v1_qwen_image_edit_2509.safetensors",
#         "prompt": "Perform an identity-preserving face swap: transfer only the face from Image 1 onto Image 2 while preserving the exact facial structure from Image 1 (eyes, nose, lips, jawline, skin texture, and proportions) without beautification or stylization. Keep Image 2 hair, body, pose, camera angle, lighting, background, and clothing unchanged. Blend edges naturally and keep a photorealistic result. face swap face from Image 1 to Image 2. swap only the face (not the hair), match the skin tone to Image 2, keep Image 2 pose and lighting.",
#         "inverted": False,
#     },
#     "head_v5_merged_rank16": {
#         "filename": "bfs_head_v5_2511_merged_version_rank_16_fp16.safetensors",
#         "prompt": "Head swap with Picture 1 as base: replace the face/head identity in Picture 1 using Picture 2 while strictly preserving Picture 2 facial identity (facial geometry, eyes, nose, lips, skin texture, and natural imperfections) with no beautification, no cartooning, and no identity drift. Preserve Picture 1 body, framing, pose, lighting, background, and scene consistency. Keep the output photorealistic with clean blending. head_swap: start with Picture 1 as the base image, keeping its lighting, environment, and background. remove the head from Picture 1 completely and replace it with the head from Picture 2, strictly preserving the hair, eye color, and nose structure of Picture 2. copy the eye direction, head rotation, and micro-expressions from Picture 1. high quality, sharp details, 4k",
#         "inverted": True,
#     },
# }
LORA_VARIANTS = {
    "face_v1": {
        "filename": "bfs_face_v1_qwen_image_edit_2509.safetensors",
        "prompt": "face swap face from Image 1 to Image 2. swap only the face (not the hair), match the skin tone to Image 2, keep Image 2 pose and lighting.",
        "inverted": False,
    },
    "head_v5_merged_rank16": {
        "filename": "bfs_head_v5_2511_merged_version_rank_16_fp16.safetensors",
        "prompt": "head_swap: start with Picture 1 as the base image, keeping its lighting, environment, and background. remove the head from Picture 1 completely and replace it with the head from Picture 2, strictly preserving the hair, eye color, and nose structure of Picture 2. copy the eye direction, head rotation, and micro-expressions from Picture 1. high quality, sharp details, 4k",
        "inverted": True,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BFS face swap CLI")
    parser.add_argument("--face", required=True, help="Path to source face image")
    parser.add_argument("--person", required=True, help="Path to target person/body image")
    parser.add_argument(
        "--variant",
        choices=LORA_VARIANTS.keys(),
        default="head_v5_merged_rank16",
        help="BFS LoRA variant",
    )
    parser.add_argument("--prompt", default=None, help="Optional custom prompt")
    parser.add_argument("--steps", type=int, default=30, help="Inference steps")
    parser.add_argument("--guidance", type=float, default=4.0, help="Guidance scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Execution device. Use 'cuda' to force GPU.",
    )
    parser.add_argument(
        "--cache-dir",
        default="./models",
        help="Model cache directory where base model and LoRA are downloaded",
    )
    parser.add_argument("--output", default=None, help="Output path. Defaults beside person image")
    return parser.parse_args()


def _resolve_device(requested: str) -> str:
    cuda_ok = torch.cuda.is_available()
    if requested == "auto":
        return "cuda" if cuda_ok else "cpu"
    if requested == "cuda" and not cuda_ok:
        raise RuntimeError(
            "CUDA was requested but is not available.\n"
            "Install GPU-enabled PyTorch and ensure NVIDIA driver is loaded."
        )
    return requested


def _ensure_lora_backend() -> None:
    if importlib.util.find_spec("peft") is None:
        raise RuntimeError(
            "Missing dependency: 'peft' is required to load LoRA weights.\n"
            "Fix: pip install peft  (or pip install -r requirements.txt)"
        )


def _load_bfs_lora_state_dict(lora_path: Path) -> dict[str, torch.Tensor]:
    state_dict = load_file(str(lora_path))

    # BFS head checkpoints include extra tensor keys (`.diff_b`, layer norm `.diff`) that
    # diffusers' Qwen LoRA converter does not consume; drop them before conversion.
    state_dict = {
        k: v
        for k, v in state_dict.items()
        if not (k.endswith(".diff_b") or k.endswith(".diff"))
    }

    # Some BFS checkpoints are in a non-diffusers LoRA format and miss `*.alpha`.
    # Diffusers' Qwen conversion expects these keys to exist.
    alpha_count = 0
    for key, down_weight in list(state_dict.items()):
        if not key.endswith(".lora_down.weight"):
            continue

        rank = down_weight.shape[0]
        alpha_tensor = torch.tensor(float(rank), dtype=down_weight.dtype)

        # Expected by qwen converter after prefix stripping.
        base_key = key
        if base_key.startswith("diffusion_model."):
            base_key = base_key[len("diffusion_model.") :]
        base_key = base_key.removesuffix(".lora_down.weight")
        alpha_key = f"{base_key}.alpha"

        if alpha_key not in state_dict:
            state_dict[alpha_key] = alpha_tensor
            alpha_count += 1

    if alpha_count > 0:
        print(f"Added {alpha_count} missing LoRA alpha keys for BFS compatibility.")

    return state_dict


def _load_pipe(cache_dir: Path, variant_filename: str, device: str) -> AutoPipelineForImage2Image:
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    base_dir = cache_dir / "Qwen-Image-Edit-2511"
    lora_dir = cache_dir / "BFS-Best-Face-Swap"
    lora_path = lora_dir / variant_filename

    load_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
    }

    pipe = AutoPipelineForImage2Image.from_pretrained(base_dir, **load_kwargs)

    state_dict = _load_bfs_lora_state_dict(lora_path)
    pipe.load_lora_weights(state_dict)

    if device == "cuda":
        # Avoid `device_map="cuda"` here: it loads the full stack on GPU at once and
        # can OOM on ~48GB cards when transformers runs its CUDA allocator warmup.
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)

    return pipe


def _call_pipe(
    pipe: AutoPipelineForImage2Image,
    image_1: Image.Image,
    image_2: Image.Image,
    prompt: str,
    steps: int,
    guidance: float,
    seed: int,
    device: str,
    progress_callback: Callable[[int, int], None] | None = None,
) -> Image.Image:
    generator = torch.Generator(device=device).manual_seed(seed)

    # Different diffusers versions may expose different argument names.
    def _step_callback(_: Any, step: int, __: Any, ___: dict[str, Any]) -> dict[str, Any]:
        if progress_callback is not None:
            progress_callback(step + 1, steps)
        return ___

    common_kwargs: dict[str, Any] = {
        "num_inference_steps": steps,
        "guidance_scale": guidance,
        "generator": generator,
    }

    if progress_callback is not None:
        common_kwargs["callback_on_step_end"] = _step_callback
        common_kwargs["callback_on_step_end_tensor_inputs"] = []

    candidates: list[dict[str, Any]] = [
        {
            "prompt": prompt,
            "image": [image_1, image_2],
            **common_kwargs,
        },
        {
            "prompt": prompt,
            "image": image_1,
            "image_2": image_2,
            **common_kwargs,
        },
        {
            "prompt": prompt,
            "image": image_1,
            "control_image": image_2,
            **common_kwargs,
        },
    ]

    last_error: Exception | None = None
    for kwargs in candidates:
        try:
            result = pipe(**kwargs)
            if hasattr(result, "images") and result.images:
                return result.images[0]
        except TypeError as exc:
            last_error = exc
            continue

    raise RuntimeError(
        "Could not run the pipeline with known argument signatures. "
        "Update diffusers or adjust `_call_pipe` argument mapping."
    ) from last_error


def _fit_on_canvas(source: Image.Image, canvas_size: tuple[int, int], fill_color: tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
    """Fit image into target canvas without changing aspect ratio."""
    canvas_w, canvas_h = canvas_size
    src_w, src_h = source.size

    scale = min(canvas_w / src_w, canvas_h / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))

    resized = source.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", canvas_size, fill_color)
    offset = ((canvas_w - new_w) // 2, (canvas_h - new_h) // 2)
    canvas.paste(resized, offset)
    return canvas


def main() -> None:
    args = parse_args()
    _ensure_lora_backend()

    face_path = Path(args.face).expanduser().resolve()
    person_path = Path(args.person).expanduser().resolve()
    cache_dir = Path(args.cache_dir).expanduser().resolve()

    if not face_path.exists() or not person_path.exists():
        raise FileNotFoundError("Face image and person image paths must exist.")

    variant = LORA_VARIANTS[args.variant]
    prompt = args.prompt or variant["prompt"]

    face_image = Image.open(face_path).convert("RGB")
    person_image = Image.open(person_path).convert("RGB")
    person_canvas_size = person_image.size
    prepared_face_image = _fit_on_canvas(face_image, person_canvas_size)
    prepared_person_image = person_image.copy()

    if variant["inverted"]:
        image_1, image_2 = prepared_person_image, prepared_face_image
    else:
        image_1, image_2 = prepared_face_image, prepared_person_image

    device = _resolve_device(args.device)
    if device == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU (CUDA unavailable or explicitly disabled)")

    pipe = _load_pipe(cache_dir=cache_dir, variant_filename=variant["filename"], device=device)
    output_image = _call_pipe(
        pipe=pipe,
        image_1=image_1,
        image_2=image_2,
        prompt=prompt,
        steps=args.steps,
        guidance=args.guidance,
        seed=args.seed,
        device=device,
    )

    # Keep final export at target dimensions without stretching.
    if output_image.size != person_canvas_size:
        output_image = _fit_on_canvas(output_image, person_canvas_size)

    if args.output:
        out_path = Path(args.output).expanduser().resolve()
    else:
        out_path = person_path.with_name(f"{person_path.stem}_bfs_swap.png")

    output_image.save(out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
