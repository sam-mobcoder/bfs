#!/usr/bin/env python3
"""FastAPI server for BFS face swap with streaming progress events."""

from __future__ import annotations

import base64
import io
import json
import os
import queue
import threading
import traceback
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image

from swap_face import (
    LORA_VARIANTS,
    _call_pipe,
    _ensure_lora_backend,
    _fit_on_canvas,
    _load_pipe,
    _resolve_device,
    _resolve_generation_size,
)

_MEMORY_MODES = frozenset({"auto", "full", "offload", "sequential"})

app = FastAPI(title="BFS Face Swap API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

PIPELINE_CACHE: dict[tuple[str, str, str, str], object] = {}
PIPELINE_CACHE_LOCK = threading.Lock()


def _sse_event(payload: dict[str, object]) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode("utf-8")


def _read_upload_image(upload: UploadFile) -> Image.Image:
    raw = upload.file.read()
    if not raw:
        raise HTTPException(status_code=400, detail=f"Empty upload: {upload.filename or 'unnamed file'}")
    try:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Invalid image upload: {upload.filename}") from exc


def _image_to_base64_png(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _to_data_url_png(image_base64: str) -> str:
    return f"data:image/png;base64,{image_base64}"


def _get_or_load_pipe(
    cache_path: Path, variant_filename: str, resolved_device: str, memory_mode: str
):
    cache_key = (str(cache_path), variant_filename, resolved_device, memory_mode)
    with PIPELINE_CACHE_LOCK:
        if cache_key in PIPELINE_CACHE:
            return PIPELINE_CACHE[cache_key]

    pipe = _load_pipe(
        cache_dir=cache_path,
        variant_filename=variant_filename,
        device=resolved_device,
        memory_mode=memory_mode,
    )
    with PIPELINE_CACHE_LOCK:
        PIPELINE_CACHE[cache_key] = pipe
    return pipe


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/swap/stream")
def swap_stream(
    face: UploadFile = File(...),
    person: UploadFile = File(...),
    variant: str = Form("head_v5_merged_rank16"),
    prompt: str | None = Form(None),
    steps: int = Form(50),
    guidance: float = Form(4.0),
    negative_prompt: str = Form(
        "blurry, lowres, bad anatomy, deformed face, extra eyes, waxy skin, over-smoothed face, artifacts"
    ),
    seed: int = Form(42),
    device: str = Form("auto"),
    max_megapixels: float = Form(2.0),
    match_person_size: bool = Form(False),
    cache_dir: str = Form("./models"),
    memory_mode: str = Form(os.environ.get("BFS_MEMORY_MODE", "auto")),
) -> StreamingResponse:
    if variant not in LORA_VARIANTS:
        raise HTTPException(status_code=400, detail=f"Invalid variant '{variant}'.")
    if steps < 1:
        raise HTTPException(status_code=400, detail="steps must be >= 1")
    if device not in {"auto", "cuda", "cpu"}:
        raise HTTPException(status_code=400, detail="device must be one of: auto, cuda, cpu")
    if memory_mode not in _MEMORY_MODES:
        raise HTTPException(
            status_code=400,
            detail=f"memory_mode must be one of: {', '.join(sorted(_MEMORY_MODES))}",
        )
    if max_megapixels <= 0:
        raise HTTPException(status_code=400, detail="max_megapixels must be > 0")

    _ensure_lora_backend()
    face_image = _read_upload_image(face)
    person_image = _read_upload_image(person)
    person_canvas_size = person_image.size
    prepared_face_image = _fit_on_canvas(face_image, person_canvas_size)
    prepared_person_image = person_image.copy()
    gen_width, gen_height = _resolve_generation_size(person_canvas_size, max_megapixels)

    selected_variant = LORA_VARIANTS[variant]
    effective_prompt = prompt or selected_variant["prompt"]

    if selected_variant["inverted"]:
        image_1, image_2 = prepared_person_image, prepared_face_image
    else:
        image_1, image_2 = prepared_face_image, prepared_person_image

    def event_generator():
        event_queue: queue.Queue[dict[str, object] | None] = queue.Queue()
        cache_path = Path(cache_dir).expanduser().resolve()
        yield _sse_event({"type": "status", "stage": "connected", "message": "Stream connected"})

        def enqueue(payload: dict[str, object]) -> None:
            event_queue.put(payload)

        def worker() -> None:
            try:
                resolved_device = _resolve_device(device)
                if resolved_device == "cuda":
                    gpu_name = torch.cuda.get_device_name(0)
                    enqueue({"type": "status", "stage": "device", "message": f"Using GPU: {gpu_name}"})
                else:
                    enqueue({"type": "status", "stage": "device", "message": "Using CPU"})

                enqueue({"type": "status", "stage": "model", "message": "Loading model and LoRA..."})
                pipe = _get_or_load_pipe(
                    cache_path=cache_path,
                    variant_filename=selected_variant["filename"],
                    resolved_device=resolved_device,
                    memory_mode=memory_mode,
                )
                enqueue({"type": "status", "stage": "inference", "message": "Running inference..."})

                last_sent_step = 0

                def on_progress(step_idx: int, total_steps: int) -> None:
                    nonlocal last_sent_step
                    if step_idx <= last_sent_step:
                        return
                    last_sent_step = step_idx
                    enqueue(
                        {
                            "type": "progress",
                            "step": step_idx,
                            "total_steps": total_steps,
                            "percent": round((step_idx / total_steps) * 100, 2),
                        }
                    )

                output_image = _call_pipe(
                    pipe=pipe,
                    image_1=image_1,
                    image_2=image_2,
                    prompt=effective_prompt,
                    steps=steps,
                    guidance=guidance,
                    negative_prompt=negative_prompt,
                    seed=seed,
                    device=resolved_device,
                    width=gen_width,
                    height=gen_height,
                    progress_callback=on_progress,
                )

                if match_person_size and output_image.size != person_canvas_size:
                    output_image = _fit_on_canvas(output_image, person_canvas_size)

                encoded = _image_to_base64_png(output_image)
                enqueue(
                    {
                        "type": "result",
                        "format": "png",
                        "width": output_image.size[0],
                        "height": output_image.size[1],
                        "image_base64": encoded,
                        "image_data_url": _to_data_url_png(encoded),
                    }
                )
                enqueue({"type": "done"})
            except Exception as exc:  # noqa: BLE001
                enqueue(
                    {
                        "type": "error",
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
            finally:
                event_queue.put(None)

        threading.Thread(target=worker, daemon=True).start()

        while True:
            item = event_queue.get()
            if item is None:
                break
            yield _sse_event(item)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/swap")
def swap_json(
    face: UploadFile = File(...),
    person: UploadFile = File(...),
    variant: str = Form("head_v5_merged_rank16"),
    prompt: str | None = Form(None),
    steps: int = Form(50),
    guidance: float = Form(4.0),
    negative_prompt: str = Form(
        "blurry, lowres, bad anatomy, deformed face, extra eyes, waxy skin, over-smoothed face, artifacts"
    ),
    seed: int = Form(42),
    device: str = Form("auto"),
    max_megapixels: float = Form(2.0),
    match_person_size: bool = Form(False),
    cache_dir: str = Form("./models"),
    memory_mode: str = Form(os.environ.get("BFS_MEMORY_MODE", "auto")),
) -> dict[str, object]:
    if variant not in LORA_VARIANTS:
        raise HTTPException(status_code=400, detail=f"Invalid variant '{variant}'.")
    if steps < 1:
        raise HTTPException(status_code=400, detail="steps must be >= 1")
    if device not in {"auto", "cuda", "cpu"}:
        raise HTTPException(status_code=400, detail="device must be one of: auto, cuda, cpu")
    if memory_mode not in _MEMORY_MODES:
        raise HTTPException(
            status_code=400,
            detail=f"memory_mode must be one of: {', '.join(sorted(_MEMORY_MODES))}",
        )
    if max_megapixels <= 0:
        raise HTTPException(status_code=400, detail="max_megapixels must be > 0")

    _ensure_lora_backend()
    face_image = _read_upload_image(face)
    person_image = _read_upload_image(person)
    person_canvas_size = person_image.size
    prepared_face_image = _fit_on_canvas(face_image, person_canvas_size)
    prepared_person_image = person_image.copy()
    gen_width, gen_height = _resolve_generation_size(person_canvas_size, max_megapixels)

    selected_variant = LORA_VARIANTS[variant]
    effective_prompt = prompt or selected_variant["prompt"]

    if selected_variant["inverted"]:
        image_1, image_2 = prepared_person_image, prepared_face_image
    else:
        image_1, image_2 = prepared_face_image, prepared_person_image

    cache_path = Path(cache_dir).expanduser().resolve()
    resolved_device = _resolve_device(device)
    pipe = _get_or_load_pipe(
        cache_path=cache_path,
        variant_filename=selected_variant["filename"],
        resolved_device=resolved_device,
        memory_mode=memory_mode,
    )
    output_image = _call_pipe(
        pipe=pipe,
        image_1=image_1,
        image_2=image_2,
        prompt=effective_prompt,
        steps=steps,
        guidance=guidance,
        negative_prompt=negative_prompt,
        seed=seed,
        device=resolved_device,
        width=gen_width,
        height=gen_height,
    )

    if match_person_size and output_image.size != person_canvas_size:
        output_image = _fit_on_canvas(output_image, person_canvas_size)

    encoded = _image_to_base64_png(output_image)
    return {
        "format": "png",
        "width": output_image.size[0],
        "height": output_image.size[1],
        "image_base64": encoded,
        "image_data_url": _to_data_url_png(encoded),
    }


if __name__ == "__main__":
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run("api:app", host=host, port=port, reload=False)
