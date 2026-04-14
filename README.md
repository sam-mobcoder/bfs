# BFS Face Swap CLI Project

This project gives you a complete Python setup to run **face/head swap** with:

- Base model: `Qwen/Qwen-Image-Edit-2511`
- LoRA repo: `Alissonerdx/BFS-Best-Face-Swap`

It provides a single script that takes **2 arguments** (face image + person image), performs the swap, and saves output in the same directory.

## 1) Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

For GPU on NVIDIA, install CUDA-enabled PyTorch wheels:

```bash
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

If the model is gated/private for your account, login first:

```bash
huggingface-cli login
```

## 2) Download models/weights

Download base model + default BFS LoRA:

```bash
python scripts/download_models.py
```

Download all listed BFS LoRA files:

```bash
python scripts/download_models.py --variant all
```

By default files go into `./models`.

## 3) Run face swap

Required:
- `--face`: source face image
- `--person`: target person/body image

```bash
python swap_face.py --face ./face.jpg --person ./person.jpg
```

Result is saved automatically beside person image as:
- `person_bfs_swap.png`

## 4) Optional arguments

```bash
python swap_face.py \
  --face ./face.jpg \
  --person ./person.jpg \
  --variant head_v5_merged_rank16 \
  --device cuda \
  --steps 30 \
  --guidance 4.0 \
  --seed 42
```

Supported variants right now:
- `face_v1`
- `head_v5_merged_rank16` (default, recommended)

## 5) Notes about image order

From the BFS model card:

- `face_v1`: standard order (`Image 1 = face`, `Image 2 = body`)
- `head_v5`: inverted order (`Image 1 = body`, `Image 2 = face`)

This project handles that automatically based on selected variant.

## 6) Hardware expectations

- CUDA GPU strongly recommended (VRAM-heavy model).
- CPU execution is possible but usually very slow.
- For `--device cuda`, the script now loads model weights directly on GPU (`device_map="cuda"`) to reduce CPU RAM pressure.

## 7) Troubleshooting

- Quick GPU check:
  ```bash
  nvidia-smi
  python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda, torch.cuda.device_count())"
  ```
- Verify memory during run:
  ```bash
  watch -n 1 nvidia-smi
  ```
- If `nvidia-smi` fails, NVIDIA driver is not loaded correctly.
- If `torch.cuda.is_available()` is `False`, your current torch build is CPU-only or CUDA runtime is missing.
- Use `--device cuda` to force GPU; the script now exits with a clear error if CUDA is unavailable.
- If you see download/auth errors: run `huggingface-cli login`.
- If pipeline call signature changes with a new `diffusers` release, update `_call_pipe` in `swap_face.py`.
- If generation quality is inconsistent, try increasing `--steps` and changing `--seed`.

## Source

- BFS model card and prompts: [Alissonerdx/BFS-Best-Face-Swap](https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap)
