#!/usr/bin/env python3
"""Download all required BFS face swap model weights into local cache directories."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

BASE_MODEL = "Qwen/Qwen-Image-Edit-2511"
LORA_REPO = "Alissonerdx/BFS-Best-Face-Swap"

# Most commonly used BFS variants from the model card.
LORA_FILES = {
    "face_v1": "bfs_face_v1_qwen_image_edit_2509.safetensors",
    "head_v3": "bfs_head_v3_qwen_image_edit_2509.safetensors",
    "head_v4": "bfs_head_v4_qwen_image_edit_2509.safetensors",
    "head_v5_original": "bfs_head_v5_2511_original.safetensors",
    "head_v5_merged_rank16": "bfs_head_v5_2511_merged_version_rank_16_fp16.safetensors",
    "head_v5_merged_rank32": "bfs_head_v5_2511_merged_version_rank_32_fp32.safetensors",
    "head_flux_klein_4b": "bfs_head_v1_flux-klein_4b.safetensors",
    "head_flux_klein_9b_rank64": "bfs_head_v1_flux-klein_9b_step3750_rank64.safetensors",
    "head_flux_klein_9b_rank128": "bfs_head_v1_flux-klein_9b_step3500_rank128.safetensors",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download BFS + base model files.")
    parser.add_argument(
        "--cache-dir",
        default="./models",
        help="Directory where Hugging Face snapshots are downloaded.",
    )
    parser.add_argument(
        "--variant",
        choices=list(LORA_FILES.keys()) + ["all"],
        default="head_v5_merged_rank16",
        help="Single LoRA variant to download, or 'all' for every listed file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading base model: {BASE_MODEL}")
    snapshot_download(repo_id=BASE_MODEL, local_dir=cache_dir / "Qwen-Image-Edit-2511")

    if args.variant == "all":
        patterns = [LORA_FILES[key] for key in LORA_FILES]
        print(f"Downloading all BFS LoRAs ({len(patterns)} files)")
    else:
        patterns = [LORA_FILES[args.variant]]
        print(f"Downloading BFS LoRA variant: {args.variant} -> {patterns[0]}")

    snapshot_download(
        repo_id=LORA_REPO,
        local_dir=cache_dir / "BFS-Best-Face-Swap",
        allow_patterns=patterns,
    )

    print("Done. Models are downloaded.")


if __name__ == "__main__":
    main()
