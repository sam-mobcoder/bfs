#!/usr/bin/env python3
"""Run BFS face swap from command line.

Usage:
    python swap_face.py --face ./face.jpg --person ./person.jpg
"""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
from typing import Any, Callable

import torch
from diffusers import AutoPipelineForImage2Image
from diffusers.loaders.lora_conversion_utils import _convert_non_diffusers_qwen_lora_to_diffusers
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
    parser.add_argument("--steps", type=int, default=50, help="Inference steps")
    parser.add_argument(
        "--guidance",
        type=float,
        default=4.0,
        help="True CFG scale (higher follows prompt more, but too high can hurt realism)",
    )
    parser.add_argument(
        "--negative-prompt",
        default="blurry, lowres, bad anatomy, deformed face, extra eyes, waxy skin, over-smoothed face, artifacts",
        help="Negative prompt to enable true CFG and suppress common face artifacts",
    )
    parser.add_argument(
        "--max-megapixels",
        type=float,
        default=2.0,
        help="Generation resolution cap in megapixels (higher can improve face detail, uses more VRAM/time)",
    )
    parser.add_argument(
        "--match-person-size",
        action="store_true",
        help="Resize output back to person image dimensions (disabled by default to avoid quality loss).",
    )
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
    parser.add_argument(
        "--memory",
        choices=["auto", "full", "offload", "sequential"],
        default=os.environ.get("BFS_MEMORY_MODE", "auto"),
        help="GPU memory strategy. 'full' keeps the pipeline on VRAM (best speed/quality on large GPUs). "
        "'offload' moves weights CPU<->GPU each step (low peak VRAM, often ~40–45GB). "
        "'auto' uses full VRAM when the GPU reports >=72GiB total memory, else offload. "
        "Override default with env BFS_MEMORY_MODE.",
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
    """Return a state dict in diffusers Qwen LoRA format, ready for `load_lora_weights`.

    BFS releases omit per-layer ``.alpha`` tensors. Diffusers' converter expects them and
    implements Kohya-style scaling ``(alpha/rank) * B @ A @ x``. Using ``alpha = rank`` is
    the usual default when alpha is absent (scale 1.0); weights are unchanged before fusion.
    """
    state_dict = load_file(str(lora_path))

    # BFS head checkpoints include extra tensor keys (`.diff_b`, layer norm `.diff`) that
    # diffusers' Qwen LoRA converter does not consume; drop them before conversion.
    state_dict = {
        k: v
        for k, v in state_dict.items()
        if not (k.endswith(".diff_b") or k.endswith(".diff"))
    }

    for key, down_weight in list(state_dict.items()):
        if not key.endswith(".lora_down.weight"):
            continue

        rank = int(down_weight.shape[0])
        base_key = key
        if base_key.startswith("diffusion_model."):
            base_key = base_key[len("diffusion_model.") :]
        base_key = base_key.removesuffix(".lora_down.weight")
        alpha_key = f"{base_key}.alpha"

        if alpha_key not in state_dict:
            state_dict[alpha_key] = torch.tensor(float(rank), dtype=down_weight.dtype)

    return _convert_non_diffusers_qwen_lora_to_diffusers(state_dict)


def _resolve_memory_mode(requested: str, device: str) -> str:
    if device != "cuda":
        return "full"
    if requested != "auto":
        return requested
    total = torch.cuda.get_device_properties(0).total_memory
    # Qwen Image Edit + LoRA often needs well under 80GiB but can exceed ~48GiB full load.
    return "full" if total >= 72 * (1024**3) else "offload"


def _move_pipeline_to_device(
    pipe: AutoPipelineForImage2Image, device: str, memory_mode: str
) -> None:
    if device != "cuda":
        pipe.to(device)
        return

    mode = _resolve_memory_mode(memory_mode, device)
    if mode == "full":
        pipe.to("cuda")
    elif mode == "sequential":
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.enable_model_cpu_offload()


def _load_pipe(
    cache_dir: Path,
    variant_filename: str,
    device: str,
    memory_mode: str = "auto",
) -> AutoPipelineForImage2Image:
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

    _move_pipeline_to_device(pipe, device, memory_mode)

    return pipe


def _call_pipe(
    pipe: AutoPipelineForImage2Image,
    image_1: Image.Image,
    image_2: Image.Image,
    prompt: str,
    steps: int,
    guidance: float,
    negative_prompt: str | None,
    seed: int,
    device: str,
    width: int | None = None,
    height: int | None = None,
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
        "true_cfg_scale": guidance,
        "negative_prompt": negative_prompt if negative_prompt is not None else " ",
        "generator": generator,
    }
    if width is not None and height is not None:
        common_kwargs["width"] = width
        common_kwargs["height"] = height

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


def _resolve_generation_size(
    source_size: tuple[int, int], max_megapixels: float, multiple: int = 32
) -> tuple[int, int]:
    src_w, src_h = source_size
    if max_megapixels <= 0:
        raise ValueError("--max-megapixels must be > 0")

    target_area = int(max_megapixels * 1_000_000)
    src_area = src_w * src_h
    if src_area <= target_area:
        width, height = src_w, src_h
    else:
        scale = (target_area / float(src_area)) ** 0.5
        width = max(multiple, int((src_w * scale) // multiple) * multiple)
        height = max(multiple, int((src_h * scale) // multiple) * multiple)

    width = max(multiple, (width // multiple) * multiple)
    height = max(multiple, (height // multiple) * multiple)
    return width, height


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

    gen_width, gen_height = _resolve_generation_size(person_canvas_size, args.max_megapixels)
    print(
        f"Generation size: {gen_width}x{gen_height} "
        f"(cap={args.max_megapixels:.2f}MP, source={person_canvas_size[0]}x{person_canvas_size[1]})"
    )

    eff_mem = _resolve_memory_mode(args.memory, device)
    if device == "cuda":
        vram_gib = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(
            f"VRAM layout: {eff_mem} (requested={args.memory}, GPU total={vram_gib:.1f} GiB). "
            "Use --memory full on large GPUs for highest throughput; --memory offload if you OOM."
        )

    pipe = _load_pipe(
        cache_dir=cache_dir,
        variant_filename=variant["filename"],
        device=device,
        memory_mode=args.memory,
    )
    output_image = _call_pipe(
        pipe=pipe,
        image_1=image_1,
        image_2=image_2,
        prompt=prompt,
        steps=args.steps,
        guidance=args.guidance,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        device=device,
        width=gen_width,
        height=gen_height,
    )

    # Optional upsize back to the input size; disabled by default to preserve sharpness.
    if args.match_person_size and output_image.size != person_canvas_size:
        output_image = _fit_on_canvas(output_image, person_canvas_size)

    if args.output:
        out_path = Path(args.output).expanduser().resolve()
    else:
        out_path = person_path.with_name(f"{person_path.stem}_bfs_swap.png")

    output_image.save(out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
