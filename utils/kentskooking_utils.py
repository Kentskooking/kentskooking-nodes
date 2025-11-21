import torch
import torch.nn.functional as F
import os
import random
import math
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import json
from datetime import datetime
import comfy.utils
import safetensors.torch

# ====================================================================================================
# MATH / WAVE FUNCTIONS
# ====================================================================================================

def triangle_wave(position, cycle_length, min_val, max_val):
    """Calculate triangle wave value at given position (Min -> Max -> Min)."""
    half_cycle = cycle_length / 2.0
    if position <= half_cycle:
        t = position / half_cycle
    else:
        t = (cycle_length - position) / half_cycle
    return min_val + (max_val - min_val) * t

def sine_wave(position, cycle_length, min_val, max_val):
    """Calculate sine wave value (Min -> Max -> Min)."""
    # Map position 0..cycle to 0..2*PI
    # We want to start at Min (like triangle), peak at Max, end at Min.
    # -cos(t) starts at -1, goes to 1, goes to -1.
    # Map -1..1 to 0..1 via 0.5 * (1 - cos(t))
    if cycle_length == 0: return min_val
    t = (position / cycle_length) * 2 * math.pi
    normalized = 0.5 * (1 - math.cos(t))
    return min_val + (max_val - min_val) * normalized

def sawtooth_wave(position, cycle_length, min_val, max_val):
    """Calculate sawtooth/ramp wave (Min -> Max)."""
    if cycle_length == 0: return min_val
    t = position / cycle_length
    return min_val + (max_val - min_val) * t

def square_wave(position, cycle_length, min_val, max_val):
    """Calculate square wave (Max for first half, Min for second half)."""
    if cycle_length == 0: return min_val
    if position < (cycle_length / 2.0):
        return max_val
    else:
        return min_val

def gaussian_bell_curve(position, cycle_length, min_val, max_val):
    """
    Calculate a bell curve peaking at the center of the cycle.
    Used for interpolation denoising (max chaos at center).
    """
    if cycle_length == 0: return min_val
    
    # Center of the cycle
    mu = cycle_length / 2.0
    # Standard deviation controls width. 
    # sigma = cycle_length / 6 ensures the curve tapers to near-zero at edges
    sigma = cycle_length / 6.0 
    
    # Gaussian formula
    exponent = -0.5 * ((position - mu) / sigma) ** 2
    # Normalized to 0..1 height (at peak exp is 1.0)
    val = math.exp(exponent)
    
    return min_val + (max_val - min_val) * val

def calculate_wave(wave_type, position, cycle_length, min_val, max_val):
    """Dispatch to specific wave function based on type."""
    if wave_type == "sine":
        return sine_wave(position, cycle_length, min_val, max_val)
    elif wave_type == "sawtooth":
        return sawtooth_wave(position, cycle_length, min_val, max_val)
    elif wave_type == "square":
        return square_wave(position, cycle_length, min_val, max_val)
    elif wave_type == "bell_curve":
        return gaussian_bell_curve(position, cycle_length, min_val, max_val)
    else:
        # Default to triangle
        return triangle_wave(position, cycle_length, min_val, max_val)

def normalize_values(values, display_min=0.0, display_max=1.0):
    """Normalize values to 0-1 range for display (used in Visualizer)."""
    arr = np.array(values)
    val_min = arr.min()
    val_max = arr.max()

    if val_max - val_min < 0.001:
        return np.ones_like(arr) * 0.5

    normalized = (arr - val_min) / (val_max - val_min)
    return normalized

# ====================================================================================================
# NOISE GENERATION
# ====================================================================================================

def generate_noise_image(width, height, seed, noise_level=8192):
    """
    Generate noise image matching Mixlab NoiseImage node.
    Creates white base with uniform random noise per channel.
    """
    random.seed(seed)
    image = Image.new("RGB", (width, height), (255, 255, 255))
    pixels = image.load()
    
    for i in range(width):
        for j in range(height):
            noise_r = random.randint(-noise_level, noise_level)
            noise_g = random.randint(-noise_level, noise_level)
            noise_b = random.randint(-noise_level, noise_level)

            r = max(0, min(pixels[i, j][0] + noise_r, 255))
            g = max(0, min(pixels[i, j][1] + noise_g, 255))
            b = max(0, min(pixels[i, j][2] + noise_b, 255))

            pixels[i, j] = (r, g, b)

    return image

def pil_to_tensor(pil_image):
    """Convert PIL image to ComfyUI tensor format [1, H, W, C]."""
    img_np = np.array(pil_image).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).unsqueeze(0)
    return img_tensor

def generate_noise_latent(latent_shape, vae, seed):
    """Generate VAE-encoded noise latent for caching."""
    batch, channels, height, width = latent_shape
    pixel_height = height * 8
    pixel_width = width * 8

    noise_image = generate_noise_image(pixel_width, pixel_height, seed, noise_level=8192)
    noise_tensor = pil_to_tensor(noise_image)
    
    # Ensure we only take RGB channels if there are more (e.g. RGBA)
    if noise_tensor.shape[-1] > 3:
         noise_tensor = noise_tensor[:,:,:,:3]
         
    noise_latent = vae.encode(noise_tensor)
    return noise_latent

def inject_noise_latent(latent, strength, vae=None, seed=0, cached_noise_latent=None):
    """
    Inject noise into latent tensor.
    Prioritizes cached_noise_latent, then VAE generation, then simple Gaussian noise.
    """
    if strength <= 0:
        return latent

    if cached_noise_latent is not None:
        return latent + cached_noise_latent * strength
    elif vae is not None:
        batch, channels, height, width = latent.shape
        pixel_height = height * 8
        pixel_width = width * 8

        noise_image = generate_noise_image(pixel_width, pixel_height, seed, noise_level=8192)
        noise_tensor = pil_to_tensor(noise_image)
        if noise_tensor.shape[-1] > 3:
             noise_tensor = noise_tensor[:,:,:,:3]
        
        noise_latent = vae.encode(noise_tensor)
        return latent + noise_latent * strength
    else:
        noise = torch.randn_like(latent) * strength
        return latent + noise

# ====================================================================================================
# CONDITIONING MODIFIERS
# ====================================================================================================

def apply_style_model_wrapper(conditioning, style_model, clip_vision_output, strength, strength_type, base_cond=None):
    """
    Apply style model to conditioning.
    Supports optional base_cond caching for performance.
    """
    if base_cond is not None:
        cond = base_cond.clone()
    else:
        cond = style_model.get_cond(clip_vision_output).flatten(start_dim=0, end_dim=1).unsqueeze(dim=0)

    if strength_type == "multiply":
        cond *= strength

    # Precompute attn_bias flag and value
    is_attn_bias = (strength_type == "attn_bias")
    attn_bias_val = strength if is_attn_bias else 1.0
    
    n = cond.shape[1]
    c_out = []
    for t in conditioning:
        (txt, keys) = t
        keys = keys.copy()
        if "attention_mask" in keys or (is_attn_bias and strength != 1.0):
            mask_ref_size = keys.get("attention_mask_img_shape", (1, 1))
            n_ref = mask_ref_size[0] * mask_ref_size[1]
            n_txt = txt.shape[1]
            mask = keys.get("attention_mask", None)
            
            if mask is None:
                mask = torch.zeros(
                    (txt.shape[0], n_txt + n_ref, n_txt + n_ref),
                    dtype=torch.float16,
                    device=txt.device
                )
            if mask.dtype == torch.bool:
                mask = torch.log(mask.to(dtype=torch.float16))
                
            new_mask = torch.zeros(
                (txt.shape[0], n_txt + n + n_ref, n_txt + n + n_ref),
                dtype=torch.float16,
                device=txt.device
            )
            
            new_mask[:, :n_txt, :n_txt] = mask[:, :n_txt, :n_txt]
            new_mask[:, :n_txt, n_txt+n:] = mask[:, :n_txt, n_txt:]
            new_mask[:, n_txt+n:, :n_txt] = mask[:, n_txt:, :n_txt]
            new_mask[:, n_txt+n:, n_txt+n:] = mask[:, n_txt:, n_txt:]
            
            attn_bias = torch.log(torch.tensor([attn_bias_val], device=txt.device))
            new_mask[:, :n_txt, n_txt:n_txt+n] = attn_bias
            new_mask[:, n_txt+n:, n_txt:n_txt+n] = attn_bias
            
            keys["attention_mask"] = new_mask
            keys["attention_mask_img_shape"] = mask_ref_size

        c_out.append([torch.cat((txt, cond), dim=1), keys])

    return c_out

def apply_controlnet_wrapper(positive, negative, control_net, control_hint, strength, start_percent, end_percent, vae=None):
    """Apply controlnet to positive and negative conditioning."""
    if strength == 0:
        return positive, negative

    cnets = {}
    out = []

    for conditioning in [positive, negative]:
        c = []
        for t in conditioning:
            d = t[1].copy()

            prev_cnet = d.get('control', None)
            if prev_cnet in cnets:
                c_net = cnets[prev_cnet]
            else:
                c_net = control_net.copy().set_cond_hint(control_hint, strength, (start_percent, end_percent), vae=vae)
                c_net.set_previous_controlnet(prev_cnet)
                cnets[prev_cnet] = c_net

            d['control'] = c_net
            d['control_apply_to_uncond'] = False
            n = [t[0], d]
            c.append(n)
        out.append(c)

    return out[0], out[1]

# ====================================================================================================
# CHECKPOINTING
# ====================================================================================================

def resolve_checkpoint_path(run_id_or_path, default_dir):
    """
    Find checkpoint file based on run ID or path.
    Handles relative paths, directory stripping, and prefix matching.
    """
    # Check if user provided a directory path or just a filename/run_id
    directory = os.path.dirname(run_id_or_path)
    filename = os.path.basename(run_id_or_path)

    # Use provided directory or default to checkpoint_dir
    checkpoint_dir = directory if directory else default_dir

    # Strip known prefixes if present
    if filename.startswith("video_ckpt_"):
        filename = filename[len("video_ckpt_"):]
    elif filename.startswith("image_ckpt_"):
        filename = filename[len("image_ckpt_"):]

    # Strip ".latent" extension if present
    if filename.endswith(".latent"):
        filename = filename[:-len(".latent")]

    # Clean run_id for display/return
    clean_run_id = filename

    # Try to find the file with either prefix
    video_path = os.path.join(checkpoint_dir, f"video_ckpt_{filename}.latent")
    image_path = os.path.join(checkpoint_dir, f"image_ckpt_{filename}.latent")

    if os.path.exists(video_path):
        return video_path, clean_run_id
    elif os.path.exists(image_path):
        return image_path, clean_run_id
    else:
        # If file not found, raise with helpful message
        raise FileNotFoundError(
            f"Checkpoint file not found.\n"
            f"Tried: {video_path}\n"
            f"  and: {image_path}"
        )

def load_checkpoint_tensors(checkpoint_path):
    """
    Load tensors and metadata from a safetensors checkpoint.
    Returns (tensors_dict, metadata_dict).
    Ensures file handle is closed and tensors are cloned to memory.
    """
    tensors = {}
    metadata = {}
    
    with safetensors.safe_open(checkpoint_path, framework="pt", device="cpu") as f:
        metadata = f.metadata() or {}
        
        # Load all keys
        for key in f.keys():
            tensors[key] = f.get_tensor(key).clone()
            
    return tensors, metadata

def save_checkpoint_file(tensors_dict, metadata, checkpoint_dir, run_id, prefix="video_ckpt_"):
    """
    Save a checkpoint file with given tensors and metadata.
    tensors_dict: Dict of tensors to save (e.g. {'latent_tensor': ...})
    """
    try:
        # Ensure required keys are present
        if "latent_format_version_0" not in tensors_dict:
            tensors_dict["latent_format_version_0"] = torch.tensor([])

        checkpoint_path = os.path.join(checkpoint_dir, f"{prefix}{run_id}.latent")
        comfy.utils.save_torch_file(tensors_dict, checkpoint_path, metadata=metadata)
        return checkpoint_path
    except Exception as e:
        print(f"⚠ Warning: Failed to save checkpoint: {e}")
        return None

def cleanup_checkpoint_files(checkpoint_dir, run_id, prefix="video_ckpt_"):
    """Delete checkpoint and preview files."""
    try:
        checkpoint_path = os.path.join(checkpoint_dir, f"{prefix}{run_id}.latent")
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            print(f"✓ Checkpoint file removed: {checkpoint_path}")

        preview_path = os.path.join(checkpoint_dir, f"{prefix}{run_id}_preview.png")
        if os.path.exists(preview_path):
            os.remove(preview_path)
            print(f"✓ Preview PNG removed: {preview_path}")
    except Exception as e:
        print(f"⚠ Warning: Failed to cleanup checkpoint files: {e}")

def generate_checkpoint_preview_image(latent_sample, vae, run_id, checkpoint_dir, prefix="video_ckpt_", prompt=None, extra_pnginfo=None, preview_type="input_latent"):
    """
    Generate and save a preview PNG from a latent sample.
    latent_sample: Single latent tensor (not batch)
    """
    try:
        decoded = vae.decode(latent_sample.unsqueeze(0))
        image = decoded[0]
        image_np = 255.0 * image.cpu().numpy()
        image_np = np.clip(image_np, 0, 255).astype(np.uint8)
        pil_image = Image.fromarray(image_np)

        metadata = PngInfo()

        if prompt is not None:
            metadata.add_text("prompt", json.dumps(prompt))
        if extra_pnginfo is not None:
            for key in extra_pnginfo:
                metadata.add_text(key, json.dumps(extra_pnginfo[key]))

        metadata.add_text("checkpoint_run_id", run_id)
        metadata.add_text("checkpoint_timestamp", datetime.now().isoformat())
        metadata.add_text("checkpoint_preview_type", preview_type)

        preview_path = os.path.join(checkpoint_dir, f"{prefix}{run_id}_preview.png")
        pil_image.save(preview_path, pnginfo=metadata, compress_level=4)
        print(f"  ✓ Preview PNG saved: {preview_path}")

    except Exception as e:
        print(f"  ⚠ Warning: Could not generate preview PNG: {e}")