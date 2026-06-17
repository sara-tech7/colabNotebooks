"""
image_generation_controlnet.py
================================
Realize Me – ControlNet + SAM + img2img Pipeline
Replaces image_generation_local.py (Pix2Pix/Scribbler).

Pipeline (exact match to sam-img2img__1_.ipynb / evaluation_pipeline.ipynb):

  Step 1 – ControlNet generation
    Base arch:   lllyasviel/control_v11p_sd15_lineart
    Weights:     finetuned, loaded from controlnet_state_dict.pt via load_state_dict()
    SD base:     runwayml/stable-diffusion-v1-5  (vanilla, no LoRA applied here)
    Scheduler:   UniPCMultistepScheduler
    steps=25  |  guidance=7.5  |  controlnet_scale=0.6–0.7
    prompt uses a `category` string supplied by the frontend (e.g. "shirt", "dress")

  Step 2 – SAM + colour assignment + blend
    SAM model:   vit_b  (sam_vit_b_01ec64.pth)
    Outer mask:  SamPredictor — 1 fg point (image center)
                 → highest-score mask from multimask_output
    Inner masks: SamAutomaticMaskGenerator filtered to ≥50% overlap with outer
    Assignment:  Only regions that CONTAIN hint pixels are recoloured.
                 Unhinted regions keep the ControlNet colour.
                 Fallback: if no region gets a hit, paint outer_mask with median hint.
    Blend:       rgb_strong_blend (alpha=0.85, mask soften σ=1.5)
                 followed by lab_blend (σ=2.0) as a soft AB-channel pass
                 (matches the two-step call sequence in run_full_pipeline)

  Step 3 – ControlNet img2img
    Pipeline:    StableDiffusionControlNetImg2ImgPipeline
    image:       LAB-blended coloured image  (init)
    control_image: original sketch           (keeps structure during refinement)
    strength=0.50  |  guidance=7.5  |  controlnet_scale=0.6–0.7  |  steps=25
    prompt uses the same `category` string supplied by the frontend

  NOTE ON LoRA
    adapter_model.safetensors (rank-4 to_q/to_v LoRA) exists but is NOT loaded
    in any notebook inference path. It is available via APPLY_LORA=true env var
    if you want to experiment, but is off by default.

Environment variables (all optional — sensible defaults shown):
  SD_BASE_MODEL          "runwayml/stable-diffusion-v1-5"
  CONTROLNET_BASE_MODEL  "lllyasviel/control_v11p_sd15_lineart"
  CONTROLNET_STATE_DICT  "../model/controlnet_state_dict.pt"
  LORA_DIR               "../model/lora"   (only used when APPLY_LORA=true)
  APPLY_LORA             "false"
  SAM_CHECKPOINT         "../model/sam_vit_b_01ec64.pth"
  SAM_MODEL_TYPE         "vit_b"
  IMG2IMG_STRENGTH       "0.50"
  CONTROLNET_CONDITIONING_SCALE  "0.65"   (kept inside the requested 0.6–0.7 range)
  DEFAULT_CATEGORY       "clothing item"  (used only if the frontend sends no category)
"""

from __future__ import annotations

import os
import logging
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, binary_closing, binary_fill_holes
from skimage.color import rgb2lab, lab2rgb

import torch
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    StableDiffusionControlNetImg2ImgPipeline,
    UniPCMultistepScheduler,
)
from segment_anything import (
    SamAutomaticMaskGenerator,
    SamPredictor,
    sam_model_registry,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

_HERE = os.path.dirname(__file__)
_MODEL_DIR = os.path.join(_HERE, "..", "model")

SD_BASE_MODEL:          str = os.environ.get("SD_BASE_MODEL",         "runwayml/stable-diffusion-v1-5")
CONTROLNET_BASE_MODEL:  str = os.environ.get("CONTROLNET_BASE_MODEL", "lllyasviel/control_v11p_sd15_lineart")
CONTROLNET_STATE_DICT:  str = os.environ.get("CONTROLNET_STATE_DICT", os.path.join(_MODEL_DIR, "controlnet_state_dict.pt"))
LORA_DIR:               str = os.environ.get("LORA_DIR",              os.path.join(_MODEL_DIR, "lora"))
APPLY_LORA:             bool = os.environ.get("APPLY_LORA", "false").lower() == "true"
SAM_CHECKPOINT:         str = os.environ.get("SAM_CHECKPOINT",        os.path.join(_MODEL_DIR, "sam_vit_b_01ec64.pth"))
SAM_MODEL_TYPE:         str = os.environ.get("SAM_MODEL_TYPE",        "vit_b")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float16 if DEVICE.type == "cuda" else torch.float32

# Inference hyperparameters — from evaluation_pipeline.ipynb / sam-img2img__1_.ipynb CONFIG
NUM_INFERENCE_STEPS:   int   = 4
GUIDANCE_SCALE:        float = 7.5
CONTROLNET_SCALE:      float = float(os.environ.get("CONTROLNET_CONDITIONING_SCALE", "0.65"))  # requested range: 0.6 – 0.7
IMG2IMG_STRENGTH:      float = float(os.environ.get("IMG2IMG_STRENGTH", "0.50"))
IMAGE_SIZE:            int   = 512

# SAM hyperparameters
SAM_POINTS_PER_SIDE:   int   = 32
SAM_IOU_THRESH:        float = 0.88
SAM_STABILITY_THRESH:  float = 0.92
SAM_MIN_MASK_AREA:     int   = 200

DEFAULT_CATEGORY: str = os.environ.get("DEFAULT_CATEGORY", "clothing item")

# ── Step 1 (ControlNet generation) prompts ─────────────────────────────────
# `category` is supplied by the frontend (e.g. "shirt", "dress", "jacket").
GENERATION_NEGATIVE_PROMPT = (
    "person, model, mannequin, hanger, rack, colored background, dark background, "
    "gradient background, shadow, low contrast, blurry edges, multiple items, "
    "text, logo, watermark, clutter"
)


def _build_generation_prompt(category: str) -> str:
    return (
        f"a pure white {category}, plain beige background, centered, "
        f"studio product photo, high quality, high contrast between garment and background"
    )


# ── Step 3 (ControlNet img2img) prompts ────────────────────────────────────
IMG2IMG_NEGATIVE_PROMPT = (
    "person, human, model, mannequin, hanger, rack, colored background, dark background, "
    "gradient background, shadow, low contrast, blurry edges, multiple items, "
    "text, logo, watermark, clutter"
)


def _build_img2img_prompt(category: str) -> str:
    return (
        f"a {category}, studio product photo, pure white background, centered, "
        f"high quality, photorealistic fabric texture"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Lazy model singletons
# ──────────────────────────────────────────────────────────────────────────────

_controlnet:         Optional[ControlNetModel]                          = None
_cn_gen_pipe:        Optional[StableDiffusionControlNetPipeline]        = None
_cn_i2i_pipe:        Optional[StableDiffusionControlNetImg2ImgPipeline] = None
_mask_generator:     Optional[SamAutomaticMaskGenerator]                = None
_predictor:          Optional[SamPredictor]                             = None


def _load_controlnet() -> ControlNetModel:
    """
    1. Load architecture from lllyasviel/control_v11p_sd15_lineart
    2. Replace weights with finetuned state_dict from controlnet_state_dict.pt
    This is the exact pattern from evaluation_pipeline.ipynb.
    """
    global _controlnet
    if _controlnet is not None:
        return _controlnet

    logger.info("⏳  Initialising ControlNet from base architecture …")
    cn = ControlNetModel.from_pretrained(
        CONTROLNET_BASE_MODEL,
        torch_dtype=DTYPE,
    )

    logger.info(f"⏳  Loading finetuned weights from {CONTROLNET_STATE_DICT} …")
    if not os.path.exists(CONTROLNET_STATE_DICT):
        raise FileNotFoundError(
            f"controlnet_state_dict.pt not found at: {CONTROLNET_STATE_DICT}\n"
            f"Set the CONTROLNET_STATE_DICT env var to its correct path."
        )
    state_dict = torch.load(CONTROLNET_STATE_DICT, map_location="cpu")
    cn.load_state_dict(state_dict)
    del state_dict

    _controlnet = cn.to(DTYPE).to(DEVICE)
    logger.info("✓  ControlNet ready")
    return _controlnet


def _load_cn_gen_pipe() -> StableDiffusionControlNetPipeline:
    """Generation pipeline: sketch → structured white/grey garment image."""
    global _cn_gen_pipe
    if _cn_gen_pipe is not None:
        return _cn_gen_pipe

    controlnet = _load_controlnet()

    logger.info("⏳  Building ControlNet generation pipeline …")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        SD_BASE_MODEL,
        controlnet=controlnet,
        torch_dtype=DTYPE,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    # Optional LoRA — off by default; not used in any notebook inference path
    if APPLY_LORA:
        lora_weights = os.path.join(LORA_DIR, "adapter_model.safetensors")
        if os.path.exists(lora_weights):
            logger.info(f"⏳  Loading UNet LoRA from {LORA_DIR} …")
            pipe.unet.load_attn_procs(LORA_DIR)
            logger.info("✓  UNet LoRA loaded")
        else:
            logger.warning(f"  APPLY_LORA=true but no adapter_model.safetensors at {LORA_DIR}")

    pipe.to(DEVICE)
    if DEVICE.type == "cuda":
        pipe.enable_xformers_memory_efficient_attention()

    _cn_gen_pipe = pipe
    logger.info("✓  ControlNet generation pipeline ready")
    return pipe


def _load_cn_i2i_pipe() -> StableDiffusionControlNetImg2ImgPipeline:
    """
    img2img pipeline that also conditions on the sketch (control_image).
    Reuses VAE / text_encoder / UNet / ControlNet already in memory.
    """
    global _cn_i2i_pipe
    if _cn_i2i_pipe is not None:
        return _cn_i2i_pipe

    base = _load_cn_gen_pipe()

    logger.info("⏳  Building ControlNet img2img pipeline …")
    pipe = StableDiffusionControlNetImg2ImgPipeline(
        vae            = base.vae,
        text_encoder   = base.text_encoder,
        tokenizer      = base.tokenizer,
        unet           = base.unet,
        controlnet     = base.controlnet,
        scheduler      = base.scheduler,
        safety_checker           = None,
        feature_extractor        = None,
        requires_safety_checker  = False,
    )
    pipe.to(DEVICE)
    if DEVICE.type == "cuda":
        pipe.enable_xformers_memory_efficient_attention()

    _cn_i2i_pipe = pipe
    logger.info("✓  ControlNet img2img pipeline ready")
    return pipe


def _load_sam() -> tuple[SamAutomaticMaskGenerator, SamPredictor]:
    global _mask_generator, _predictor
    if _mask_generator is not None:
        return _mask_generator, _predictor

    logger.info(f"⏳  Loading SAM ({SAM_MODEL_TYPE}) from {SAM_CHECKPOINT} …")
    if not os.path.exists(SAM_CHECKPOINT):
        raise FileNotFoundError(
            f"SAM checkpoint not found at: {SAM_CHECKPOINT}\n"
            f"Download sam_vit_b_01ec64.pth from "
            f"https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth\n"
            f"or set the SAM_CHECKPOINT env var."
        )
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
    sam.to(DEVICE)
    sam.eval()

    _mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side         = SAM_POINTS_PER_SIDE,
        pred_iou_thresh         = SAM_IOU_THRESH,
        stability_score_thresh  = SAM_STABILITY_THRESH,
        min_mask_region_area    = SAM_MIN_MASK_AREA,
    )
    _predictor = SamPredictor(sam)
    logger.info("✓  SAM ready")
    return _mask_generator, _predictor


# ──────────────────────────────────────────────────────────────────────────────
# Step 1 helpers
# ──────────────────────────────────────────────────────────────────────────────

def _preprocess_sketch(sketch: Image.Image) -> tuple[Image.Image, np.ndarray]:
    """
    Returns:
      sketch_pil : RGB PIL Image 512×512 (for pipeline calls)
      sketch_np  : float32 (512,512) grayscale [0,1] (for SAM foreground point)
    """
    sketch_pil  = sketch.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
    sketch_gray = np.array(sketch_pil.convert("L"), dtype=np.float32) / 255.0
    return sketch_pil, sketch_gray


# ──────────────────────────────────────────────────────────────────────────────
# Step 2 helpers — SAM
# ──────────────────────────────────────────────────────────────────────────────

def _sketch_foreground_point(sketch_gray: np.ndarray) -> np.ndarray:
    """Median (x,y) of dark sketch pixels. Falls back to image centre."""
    line = sketch_gray < 0.6
    ys, xs = np.where(line)
    if len(xs) < 10:
        H, W = sketch_gray.shape
        return np.array([W // 2, H // 2])
    return np.array([int(np.median(xs)), int(np.median(ys))])


def _get_outer_mask(
    image_np: np.ndarray,           # uint8 RGB
    sketch_gray: Optional[np.ndarray],
    predictor: SamPredictor,
) -> np.ndarray:
    """
    Prompted SAM: single center-point foreground prompt.
    Uses the image center as the seed point (cx, cy), matching
    `get_outer_mask` in sam_segmentation.ipynb.
    Returns bool (H,W).
    """
    H, W = image_np.shape[:2]
    predictor.set_image(image_np)

    cx, cy = W // 2, H // 2

    masks, scores, _ = predictor.predict(
        point_coords=np.array([[cx, cy]]),
        point_labels=np.array([1]),
        multimask_output=True,
    )

    outer = masks[np.argmax(scores)].astype(bool)
    outer = binary_fill_holes(binary_closing(outer, iterations=3))
    return outer


def _filter_masks_to_outer(
    all_masks: list[dict],
    outer_mask: np.ndarray,
) -> list[dict]:
    """
    Keep only auto masks with ≥50% overlap with outer_mask, clipped to it.
    Sorted by area ascending (small first). Matches `filter_masks_to_outer`.
    """
    filtered = []
    for m in all_masks:
        seg  = m["segmentation"].astype(bool)
        area = seg.sum()
        if area == 0:
            continue
        if (seg & outer_mask).sum() / area >= 0.5:
            m               = dict(m)
            m["segmentation"] = seg & outer_mask
            m["area"]       = int(m["segmentation"].sum())
            if m["area"] >= SAM_MIN_MASK_AREA:
                filtered.append(m)
    return sorted(filtered, key=lambda x: x["area"])


# ──────────────────────────────────────────────────────────────────────────────
# Step 2 helpers — colour hints (from tldraw colour_hints.png)
# ──────────────────────────────────────────────────────────────────────────────

def _parse_color_hints(
    color_hints_img: Image.Image,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert user colour-hint canvas to:
      color_map : (512,512,3) float32 [0,1]  — RGB at painted pixels
      hint_mask : (512,512,1) float32 [0,1]  — 1 where painted

    Painted = HSV saturation > 0.15 (excludes white/grey canvas).
    """
    img = (
        color_hints_img
        .resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
        .convert("RGB")
    )
    img_np = np.array(img, dtype=np.float32) / 255.0

    hsv       = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    painted   = (hsv[:, :, 1].astype(np.float32) / 255.0) > 0.15

    color_map            = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
    color_map[painted]   = img_np[painted]
    hint_mask            = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 1), dtype=np.float32)
    hint_mask[painted, 0] = 1.0
    return color_map, hint_mask


# ──────────────────────────────────────────────────────────────────────────────
# Step 2 helpers — colour assignment
# ──────────────────────────────────────────────────────────────────────────────

def _assign_colors(
    masks: list[dict],
    color_map: np.ndarray,
    hint_mask: np.ndarray,
    image_np: np.ndarray,       # uint8 ControlNet output
    outer_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Assign hint colours to SAM regions.

    Key behavioural differences from earlier version (matches sam-img2img__1_.ipynb):
    - colored_np starts as a COPY of the ControlNet float image (not all-white)
    - Only regions that CONTAIN hint pixels are recoloured
    - Unhinted regions are left as-is (keep ControlNet colour)
    - fill_unhinted_with_nearest is False (the production default)
    - filled_mask tracks which pixels were actually recoloured (used for blend weight)

    Returns:
      colored_np   : (H,W,3) float32 [0,1]
      filled_mask  : (H,W,1) float32 [0,1]  — 1 where colour was changed
    """
    H, W        = IMAGE_SIZE, IMAGE_SIZE
    img_float   = image_np.astype(float) / 255.0
    colored_np  = img_float.copy()                          # ← copy, not white
    filled_mask = np.zeros((H, W), dtype=bool)

    hint_points = np.argwhere(hint_mask[:, :, 0] > 0.05)
    has_hints   = len(hint_points) > 0

    # If no internal masks at all, treat outer_mask as one big region
    if not masks and outer_mask.any():
        masks = [{"segmentation": outer_mask, "area": int(outer_mask.sum())}]

    for m in masks:
        seg = m["segmentation"].astype(bool) & outer_mask
        if seg.sum() == 0 or not has_hints:
            continue

        region_hint_mask = seg & (hint_mask[:, :, 0] > 0.05)
        if region_hint_mask.any():
            # Median of hint pixels inside this region
            region_color        = np.median(color_map[region_hint_mask], axis=0)
            colored_np[seg]     = region_color
            filled_mask[seg]    = True
        # else: unhinted region — keep ControlNet colour, don't set filled_mask

    # Fallback: no region got a hit → apply median hint colour to entire outer mask
    if has_hints and not filled_mask.any() and outer_mask.any():
        median_color             = np.median(color_map[hint_mask[:, :, 0] > 0.05], axis=0)
        colored_np[outer_mask]   = median_color
        filled_mask[outer_mask]  = True

    return colored_np, filled_mask[..., None].astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Step 2 helpers — blending
# ──────────────────────────────────────────────────────────────────────────────

def _rgb_strong_blend(
    structured_np: np.ndarray,
    colored_np: np.ndarray,
    region_mask: np.ndarray,    # (H,W,1) float32
    alpha: float = 0.85,
) -> np.ndarray:
    """
    Strong RGB blend inside recoloured regions.
    Matches `rgb_strong_blend` in sam-img2img__1_.ipynb.
    """
    mask = gaussian_filter(region_mask[:, :, 0].astype(np.float32), sigma=1.5)
    mask = np.clip(mask, 0, 1)[:, :, None]

    blended = (1.0 - alpha) * structured_np + alpha * colored_np
    out     = (1.0 - mask) * structured_np + mask * blended
    return np.clip(out, 0, 1)


def _lab_blend(
    structured_np: np.ndarray,
    colored_np: np.ndarray,
    region_mask: np.ndarray,    # (H,W,1) float32  — the filled_mask
    blend_sigma: float = 2.0,
) -> np.ndarray:
    """
    Replace AB (colour) channels of structured_np with those from colored_np,
    weighted by region_mask. L channel unchanged (preserves all shading).
    Matches `lab_blend` in sam-img2img__1_.ipynb (sigma=2.0, no ×4 amplifier).
    """
    struct_lab = rgb2lab(structured_np.clip(0, 1))
    color_lab  = rgb2lab(colored_np.clip(0, 1))

    weight = region_mask[:, :, 0].astype(float)
    if blend_sigma > 0:
        weight = gaussian_filter(weight, sigma=blend_sigma)
    weight = np.clip(weight, 0, 1)

    blended_lab             = struct_lab.copy()
    blended_lab[:, :, 1]    = weight * color_lab[:, :, 1] + (1 - weight) * struct_lab[:, :, 1]
    blended_lab[:, :, 2]    = weight * color_lab[:, :, 2] + (1 - weight) * struct_lab[:, :, 2]

    return np.clip(lab2rgb(blended_lab).astype(np.float32), 0, 1)


# ──────────────────────────────────────────────────────────────────────────────
# Full colour pipeline (Steps 2a – 2d)
# ──────────────────────────────────────────────────────────────────────────────

def _apply_color_pipeline(
    controlnet_output: Image.Image,
    color_hints:       Image.Image,
    sketch_gray:       np.ndarray,
) -> Image.Image:
    """
    SAM segmentation → colour assignment → RGB blend → LAB blend.
    Returns blended PIL Image ready for img2img.
    """
    mask_gen, predictor = _load_sam()

    cn_np   = np.array(controlnet_output.convert("RGB"))   # uint8
    cn_float = cn_np.astype(np.float32) / 255.0

    color_map, hint_mask = _parse_color_hints(color_hints)

    logger.info("  [SAM] Outer mask (prompted) …")
    outer_mask = _get_outer_mask(cn_np, sketch_gray, predictor)

    logger.info("  [SAM] Internal masks (automatic) …")
    all_masks      = mask_gen.generate(cn_np)
    internal_masks = _filter_masks_to_outer(all_masks, outer_mask)
    logger.info(f"  [SAM] {len(internal_masks)} internal region(s)")

    logger.info("  Assigning colours …")
    colored_np, filled_mask = _assign_colors(
        internal_masks, color_map, hint_mask, cn_np, outer_mask
    )

    logger.info("  RGB blend (alpha=0.85) …")
    blended = _rgb_strong_blend(cn_float, colored_np, filled_mask, alpha=0.85)

    logger.info("  LAB blend (σ=2.0) …")
    blended = _lab_blend(cn_float, blended, filled_mask, blend_sigma=2.0)

    logger.info("  White background (Step 4) …")
    blended[~outer_mask] = 1.0

    return Image.fromarray((blended * 255).clip(0, 255).astype(np.uint8))


# ──────────────────────────────────────────────────────────────────────────────
# Step 1 — ControlNet generation
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _run_controlnet(
    sketch_pil: Image.Image,
    seed: Optional[int],
    category: str = DEFAULT_CATEGORY,
) -> Image.Image:
    pipe      = _load_cn_gen_pipe()
    generator = torch.Generator(device=DEVICE).manual_seed(seed) if seed is not None else None

    result = pipe(
        prompt                         = _build_generation_prompt(category),
        negative_prompt                = GENERATION_NEGATIVE_PROMPT,
        image                          = sketch_pil,
        num_inference_steps            = NUM_INFERENCE_STEPS,
        guidance_scale                 = GUIDANCE_SCALE,
        controlnet_conditioning_scale  = CONTROLNET_SCALE,
        generator                      = generator,
        height                         = IMAGE_SIZE,
        width                          = IMAGE_SIZE,
    )
    return result.images[0]


# ──────────────────────────────────────────────────────────────────────────────
# Step 3 — ControlNet img2img
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _run_controlnet_img2img(
    blended_pil: Image.Image,
    sketch_pil:  Image.Image,
    seed:        Optional[int],
    strength:    float,
    category:    str = DEFAULT_CATEGORY,
) -> Image.Image:
    """
    image         = blended coloured garment  (starting point)
    control_image = original sketch           (structural lock)
    Matches `run_img2img` in sam-img2img__1_.ipynb.
    """
    pipe      = _load_cn_i2i_pipe()
    generator = torch.Generator(device=DEVICE).manual_seed(seed) if seed is not None else None

    # sketch must be RGB for ControlNet conditioning
    sketch_rgb = sketch_pil.convert("RGB")

    result = pipe(
        prompt                         = _build_img2img_prompt(category),
        negative_prompt                = IMG2IMG_NEGATIVE_PROMPT,
        image                          = blended_pil,       # init image
        control_image                  = sketch_rgb,        # structural control
        strength                       = strength,
        guidance_scale                 = GUIDANCE_SCALE,
        controlnet_conditioning_scale  = CONTROLNET_SCALE,
        num_inference_steps            = NUM_INFERENCE_STEPS,
        generator                      = generator,
    )
    return result.images[0]


# ──────────────────────────────────────────────────────────────────────────────
# Public API — called by app.py
# ──────────────────────────────────────────────────────────────────────────────

def generate_image_controlnet(
    sketch:           Image.Image,
    color_hints:      Image.Image,
    seed:             Optional[int] = None,
    img2img_strength: float         = IMG2IMG_STRENGTH,
    category:         str           = DEFAULT_CATEGORY,
) -> Image.Image:
    """
    Full pipeline: sketch + colour hints → final realistic coloured garment.

    Parameters
    ----------
    sketch          PIL Image — outline drawing from tldraw (outline mode).
                    Black/grey lines on white background.
    color_hints     PIL Image — colour strokes from tldraw (colour mode).
                    Spatially aligned with sketch, same canvas size.
    seed            Optional int for reproducibility.
    img2img_strength float [0,1].  0.50 recommended.
    category        Garment category string sent by the frontend
                    (e.g. "shirt", "dress", "jacket"). Used to build the
                    text prompts for both the ControlNet generation step
                    and the ControlNet img2img refinement step. Defaults
                    to DEFAULT_CATEGORY ("clothing item") if not provided,
                    so existing callers that don't pass it keep working.

    Returns
    -------
    PIL Image — 512×512 final image.
    """
    sketch_pil, sketch_gray = _preprocess_sketch(sketch)

    logger.info("── Step 1: ControlNet generation ──────────────────────────")
    cn_output = _run_controlnet(sketch_pil, seed, category=category)
    logger.info(f"  ControlNet output: {cn_output.size}")

    logger.info("── Step 2: SAM + colour + blend ─────────────────────────")
    blended_pil = _apply_color_pipeline(cn_output, color_hints, sketch_gray)
    logger.info(f"  Blended: {blended_pil.size}")

    logger.info("── Step 3: ControlNet img2img ───────────────────────────")
    final = _run_controlnet_img2img(blended_pil, sketch_pil, seed, img2img_strength, category=category)
    logger.info(f"  Final: {final.size}")

    return final