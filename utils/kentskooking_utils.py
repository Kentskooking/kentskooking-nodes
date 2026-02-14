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

def calculate_explorer_strengths(iteration_idx, iteration_count, curve_type="linear",
                                positive_a_start=0.0, positive_a_end=1.0,
                                positive_b_start=1.0, positive_b_end=0.0):
    """
    Calculate crossfade strengths for ExplorerConditioningSampler.
    A rises from 0->1 while B falls from 1->0 across the traversal.
    """
    if iteration_count <= 1:
        return 1.0, 1.0

    t = iteration_idx / max(1, iteration_count - 1)

    t_curve = apply_explorer_curve(t, curve_type)

    strength_a = float(positive_a_start + (positive_a_end - positive_a_start) * t_curve)
    strength_b = float(positive_b_start + (positive_b_end - positive_b_start) * t_curve)
    return strength_a, strength_b

def apply_explorer_curve(t, curve_type="linear"):
    """
    Apply explorer curve shaping to a normalized value t in [0, 1].
    """
    t = max(0.0, min(1.0, float(t)))
    if curve_type == "sine":
        # Smooth monotonic 0->1 using half cosine.
        return 0.5 - 0.5 * math.cos(math.pi * t)
    if curve_type == "ease_in_out":
        # Smoothstep
        return t * t * (3.0 - 2.0 * t)
    return t

def calculate_explorer_path_strengths(
    iteration_idx,
    iteration_count,
    curve_type="linear",
    positive_a_start=0.0,
    positive_a_end=1.0,
    positive_b_start=1.0,
    positive_b_end=0.0,
    positive_c_start=0.0,
    positive_c_end=0.0,
    has_positive_c=False,
    loop_video=False,
):
    """
    Calculate explorer strengths across a segmented traversal path.

    Path layout:
    - Without positive C: A->B (single segment; legacy behavior)
    - With positive C: A->B, then B->C
    - With positive C and loop_video: A->B, B->C, C->A

    Segment endpoints are defined by these anchor states:
    S0: (A_start, B_start, C_start)
    S1: (A_end,   B_end,   C_start)   # after A->B
    S2: (A_end,   B_start, C_end)     # after B->C
    S3: S0                              # after C->A when looping
    """
    if iteration_count <= 1:
        return 1.0, 1.0, (1.0 if has_positive_c else 0.0)

    if not has_positive_c:
        strength_a, strength_b = calculate_explorer_strengths(
            iteration_idx,
            iteration_count,
            curve_type,
            positive_a_start,
            positive_a_end,
            positive_b_start,
            positive_b_end,
        )
        return strength_a, strength_b, 0.0

    state0 = {
        "a": float(positive_a_start),
        "b": float(positive_b_start),
        "c": float(positive_c_start),
    }
    state1 = {
        "a": float(positive_a_end),
        "b": float(positive_b_end),
        "c": float(positive_c_start),
    }
    state2 = {
        "a": float(positive_a_end),
        "b": float(positive_b_start),
        "c": float(positive_c_end),
    }

    states = [state0, state1, state2]
    if loop_video:
        states.append(state0.copy())

    segment_count = len(states) - 1
    global_t = iteration_idx / max(1, iteration_count - 1)
    segment_pos = global_t * segment_count
    segment_idx = min(int(segment_pos), segment_count - 1)
    local_t = segment_pos - segment_idx
    if iteration_idx >= iteration_count - 1:
        local_t = 1.0

    t_curve = apply_explorer_curve(local_t, curve_type)
    start_state = states[segment_idx]
    end_state = states[segment_idx + 1]

    strength_a = float(start_state["a"] + (end_state["a"] - start_state["a"]) * t_curve)
    strength_b = float(start_state["b"] + (end_state["b"] - start_state["b"]) * t_curve)
    strength_c = float(start_state["c"] + (end_state["c"] - start_state["c"]) * t_curve)
    return strength_a, strength_b, strength_c

def build_explorer_conditioning(conditioning_a, conditioning_b, strength_a, strength_b):
    """
    Build weighted concat conditioning for explorer traversal.
    Follows native ConditioningConcat semantics by using the first item from
    conditioning_b and concatenating it onto every entry in conditioning_a.
    """
    return build_explorer_conditioning_multi(
        [conditioning_a, conditioning_b],
        [strength_a, strength_b],
    )

def build_explorer_conditioning_multi(conditionings, strengths):
    """
    Build weighted concat conditioning for 2+ branches.
    Branch 0 is treated as the base list. Branches 1..N follow native
    ConditioningConcat behavior by using only each branch's first cond entry.
    """
    if len(conditionings) == 0:
        raise ValueError("conditionings is empty.")
    if len(conditionings) != len(strengths):
        raise ValueError(
            f"conditionings/strengths length mismatch: {len(conditionings)} vs {len(strengths)}"
        )

    for idx, cond in enumerate(conditionings):
        if cond is None:
            raise ValueError(f"conditioning at index {idx} is None.")
        if len(cond) == 0:
            raise ValueError(f"conditioning at index {idx} is empty.")
        if idx > 0 and len(cond) > 1:
            print(
                f"Warning: ExplorerConditioningSampler conditioning index {idx} has more than 1 cond; only the first one is applied."
            )

    base_conditioning = conditionings[0]
    branch_refs = []
    for idx, cond in enumerate(conditionings):
        if idx == 0:
            branch_refs.append({"index": idx, "cond": None, "pooled": None})
            continue
        branch_refs.append({
            "index": idx,
            "cond": cond[0][0],
            "pooled": cond[0][1].get("pooled_output", None),
        })

    out = []
    for entry in base_conditioning:
        base_cond = entry[0]
        base_meta = entry[1].copy()
        base_pooled = base_meta.get("pooled_output", None)

        cond_tensors = [base_cond]
        pooled_tensors = [base_pooled]

        for ref in branch_refs[1:]:
            cond_tensors.append(ref["cond"])
            pooled_tensors.append(ref["pooled"])

        embedding_size = cond_tensors[0].shape[2]
        for tensor_idx, cond_tensor in enumerate(cond_tensors):
            if cond_tensor.shape[2] != embedding_size:
                raise ValueError(
                    f"Conditioning embedding size mismatch at branch {tensor_idx}: "
                    f"expected hidden size {embedding_size}, got tensor shape {cond_tensor.shape}."
                )

        target_batch = max(t.shape[0] for t in cond_tensors)
        aligned_conds = []
        for tensor_idx, cond_tensor in enumerate(cond_tensors):
            if cond_tensor.shape[0] == target_batch:
                aligned_conds.append(cond_tensor)
            elif cond_tensor.shape[0] == 1:
                aligned_conds.append(cond_tensor.repeat(target_batch, 1, 1))
            else:
                raise ValueError(
                    f"Conditioning batch mismatch at branch {tensor_idx}: "
                    f"cannot align batch {cond_tensor.shape[0]} to {target_batch}."
                )

        weighted_parts = []
        for idx, cond_tensor in enumerate(aligned_conds):
            weighted_parts.append(torch.mul(cond_tensor, float(strengths[idx])))
        merged = torch.cat(weighted_parts, dim=1)

        pooled_output = None
        for idx, pooled_tensor in enumerate(pooled_tensors):
            if pooled_tensor is None:
                continue
            pooled_current = pooled_tensor
            if pooled_current.dim() > 0 and pooled_current.shape[0] != target_batch:
                if pooled_current.shape[0] == 1:
                    repeats = [target_batch] + [1] * (pooled_current.dim() - 1)
                    pooled_current = pooled_current.repeat(*repeats)
                else:
                    raise ValueError(
                        f"Pooled output batch mismatch at branch {idx}: "
                        f"cannot align batch {pooled_current.shape[0]} to {target_batch}."
                    )

            weighted_pooled = torch.mul(pooled_current, float(strengths[idx]))
            pooled_output = weighted_pooled if pooled_output is None else (pooled_output + weighted_pooled)

        if pooled_output is not None:
            base_meta["pooled_output"] = pooled_output
        elif "pooled_output" in base_meta:
            del base_meta["pooled_output"]

        out.append([merged, base_meta])

    return out

def _build_explorer_conditioning_legacy_impl(conditioning_a, conditioning_b, strength_a, strength_b):
    """
    Legacy pairwise implementation retained for reference.
    """
    if len(conditioning_a) == 0:
        raise ValueError("conditioning_a is empty.")
    if len(conditioning_b) == 0:
        raise ValueError("conditioning_b is empty.")
    if len(conditioning_b) > 1:
        print("Warning: ExplorerConditioningSampler conditioning_b contains more than 1 cond, only the first one is applied.")

    cond_b = conditioning_b[0][0]
    pooled_b = conditioning_b[0][1].get("pooled_output", None)
    out = []

    for entry in conditioning_a:
        cond_a = entry[0]
        if cond_a.shape[2] != cond_b.shape[2]:
            raise ValueError(
                f"Conditioning embedding size mismatch: A={cond_a.shape}, B={cond_b.shape}. "
                "Both branches must come from compatible text encoders."
            )

        cond_b_current = cond_b
        cond_a_current = cond_a
        if cond_a_current.shape[0] != cond_b_current.shape[0]:
            if cond_b_current.shape[0] == 1:
                cond_b_current = cond_b_current.repeat(cond_a_current.shape[0], 1, 1)
            elif cond_a_current.shape[0] == 1:
                cond_a_current = cond_a_current.repeat(cond_b_current.shape[0], 1, 1)
            else:
                raise ValueError(
                    f"Conditioning batch mismatch: A={cond_a_current.shape[0]}, B={cond_b_current.shape[0]}."
                )

        weighted_a = torch.mul(cond_a_current, strength_a)
        weighted_b = torch.mul(cond_b_current, strength_b)
        merged = torch.cat((weighted_a, weighted_b), dim=1)

        meta = entry[1].copy()
        pooled_a = meta.get("pooled_output", None)
        pooled_output = None

        if pooled_a is not None and pooled_b is not None:
            pooled_output = torch.mul(pooled_a, strength_a) + torch.mul(pooled_b, strength_b)
        elif pooled_a is not None:
            pooled_output = torch.mul(pooled_a, strength_a)
        elif pooled_b is not None:
            pooled_output = torch.mul(pooled_b, strength_b)

        if pooled_output is not None:
            meta["pooled_output"] = pooled_output
        elif "pooled_output" in meta:
            del meta["pooled_output"]

        out.append([merged, meta])

    return out

def repeat_conditioning_to_batch(conditioning, target_batch):
    """
    Repeat conditioning batch dimension to target size when source batch is 1.
    Tensor metadata entries with matching batch dimension are repeated as well.
    """
    if target_batch <= 0:
        raise ValueError("target_batch must be >= 1")

    out = []
    for entry in conditioning:
        cond = entry[0]
        meta = entry[1].copy()
        source_batch = cond.shape[0]

        if source_batch == target_batch:
            cond_out = cond
        elif source_batch == 1:
            cond_out = cond.repeat(target_batch, 1, 1)
        else:
            raise ValueError(
                f"Cannot repeat conditioning batch {source_batch} to target batch {target_batch}."
            )

        for key, value in list(meta.items()):
            if isinstance(value, torch.Tensor) and value.dim() > 0 and value.shape[0] == source_batch:
                if source_batch == target_batch:
                    continue
                if source_batch == 1:
                    repeats = [target_batch] + [1] * (value.dim() - 1)
                    meta[key] = value.repeat(*repeats)
                else:
                    raise ValueError(
                        f"Cannot repeat metadata tensor '{key}' batch {source_batch} to {target_batch}."
                    )

        out.append([cond_out, meta])

    return out

def stack_conditioning_batches(conditioning_batches):
    """
    Stack multiple conditioning lists along the batch dimension while preserving order.
    Expects conditioning_batches as [conditioning_for_sample0, conditioning_for_sample1, ...].
    """
    if len(conditioning_batches) == 0:
        raise ValueError("conditioning_batches is empty.")

    expected_len = len(conditioning_batches[0])
    for idx, cond in enumerate(conditioning_batches):
        if len(cond) != expected_len:
            raise ValueError(
                f"Conditioning list length mismatch at index {idx}: "
                f"expected {expected_len}, got {len(cond)}."
            )

    out = []
    for entry_idx in range(expected_len):
        cond_tensors = [batch[entry_idx][0] for batch in conditioning_batches]
        merged_cond = torch.cat(cond_tensors, dim=0)

        metas = [batch[entry_idx][1] for batch in conditioning_batches]
        meta_out = metas[0].copy()
        source_batches = [tensor.shape[0] for tensor in cond_tensors]

        for key, value in list(meta_out.items()):
            if not isinstance(value, torch.Tensor) or value.dim() == 0:
                continue

            cat_values = []
            can_cat = True
            for meta_idx, meta in enumerate(metas):
                v = meta.get(key, None)
                if not isinstance(v, torch.Tensor) or v.dim() == 0:
                    can_cat = False
                    break
                if v.shape[0] != source_batches[meta_idx]:
                    can_cat = False
                    break
                cat_values.append(v)

            if can_cat:
                meta_out[key] = torch.cat(cat_values, dim=0)

        out.append([merged_cond, meta_out])

    return out

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
