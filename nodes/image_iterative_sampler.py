import torch
import torch.nn.functional as F
import comfy.samplers
import comfy.utils
import comfy.model_management
import nodes
import os
import random
import numpy as np
from PIL import Image
from ..utils.kentskooking_utils import (
    generate_noise_latent,
    inject_noise_latent,
    apply_style_model_wrapper,
    apply_controlnet_wrapper,
    save_checkpoint_file,
    cleanup_checkpoint_files,
    generate_checkpoint_preview_image
)

# Import IPAdapter execute function
try:
    import custom_nodes.ComfyUI_IPAdapter_plus.IPAdapterPlus as IPAdapterModule
    ipadapter_execute = IPAdapterModule.ipadapter_execute
    IPADAPTER_AVAILABLE = True
    print("✓ IPAdapter Plus imported successfully")
except ImportError as e:
    IPADAPTER_AVAILABLE = False
    print(f"⚠ Warning: IPAdapter ImportError - {e}")
except Exception as e:
    IPADAPTER_AVAILABLE = False
    print(f"⚠ Warning: IPAdapter import failed - {type(e).__name__}: {e}")


class ImageIterativeSampler:
    """
    Iterative sampler for progressive refinement of a single latent image.
    Each cycle uses incrementally more denoising steps and switches CLIP/IPAdapter images.
    Output is a batch of all iterations showing progressive refinement.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "wave_config": ("WAVE_CONFIG",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "add_noise": ("BOOLEAN", {"default": True, "label_on": "enable", "label_off": "disable"}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "noise_injection_strength": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "lock_injection_seed": ("BOOLEAN", {"default": False, "label_on": "locked", "label_off": "varying"}),
                "feedback_mode": (["none", "previous_iteration"], {"default": "none"}),
            },
            "optional": {
                "vae": ("VAE", {"tooltip": "Required when zoom is enabled (zoom_min != zoom_max) for pixel-space zoom processing. Optional for noise injection and checkpoint preview."} ),
                "checkpoint_config": ("CHECKPOINT_CONFIG",),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent_batch",)
    FUNCTION = "process_image"
    CATEGORY = "kentskooking/sampling"

    def calculate_wave_values(self, iteration_in_cycle, config, iterations_per_cycle):
        """
        Calculate zoom/clip strength for the current iteration.
        Zoom uses exponential formula (constant rate per iteration).
        Clip strength uses linear interpolation within cycle.
        """
        # Exponential zoom: zoom_rate (or zoom_max) is the per-iteration multiplier (e.g., 1.05 = 5%)
        # +1 so first iteration applies zoom (not 1.0x)
        zoom_rate = config.get("zoom_rate", config.get("zoom_max", 1.0))
        zoom = zoom_rate ** (iteration_in_cycle + 1)

        # Linear interpolation for clip strength within cycle
        if iterations_per_cycle <= 1:
            t = 0.0
        else:
            t = iteration_in_cycle / (iterations_per_cycle - 1)
        clip_strength = config.get("clip_strength_min", 1.0) + (config.get("clip_strength_max", 1.0) - config.get("clip_strength_min", 1.0)) * t

        return zoom, clip_strength

    def calculate_ipadapter_wave_values(self, iteration_idx, config, num_iterations):
        """Calculate IPAdapter values with linear interpolation across iterations."""
        if num_iterations <= 1:
            t = 0.0
        else:
            t = iteration_idx / (num_iterations - 1)

        weight = config.get("ipadapter_weight_min", 1.0) + (config.get("ipadapter_weight_max", 1.0) - config.get("ipadapter_weight_min", 1.0)) * t
        start_at = config.get("ipadapter_start_at_min", 0.0) + (config.get("ipadapter_start_at_max", 0.0) - config.get("ipadapter_start_at_min", 0.0)) * t
        end_at = config.get("ipadapter_end_at_min", 1.0) + (config.get("ipadapter_end_at_max", 1.0) - config.get("ipadapter_end_at_min", 1.0)) * t

        return weight, start_at, end_at

    def calculate_controlnet_wave_values(self, iteration_idx, config, num_iterations):
        """Calculate ControlNet values with linear interpolation across iterations."""
        if num_iterations <= 1:
            t = 0.0
        else:
            t = iteration_idx / (num_iterations - 1)

        strength = config.get("controlnet_strength_min", 1.0) + (config.get("controlnet_strength_max", 1.0) - config.get("controlnet_strength_min", 1.0)) * t
        start_at = config.get("controlnet_start_at_min", 0.0) + (config.get("controlnet_start_at_max", 0.0) - config.get("controlnet_start_at_min", 0.0)) * t
        end_at = config.get("controlnet_end_at_min", 1.0) + (config.get("controlnet_end_at_max", 1.0) - config.get("controlnet_end_at_min", 1.0)) * t

        return strength, start_at, end_at

    def zoom_latent_pixel_space(self, latent, zoom_factor, vae, base_pixel_image=None):
        """
        Performs zoom in pixel space to avoid latent corruption.
        Process: VAE decode → bicubic resize → center crop → VAE encode

        latent: (B, C, H, W) latent tensor (used if base_pixel_image not provided)
        zoom_factor: float > 0 (zoom multiplier)
        vae: VAE model for decode/encode
        base_pixel_image: Optional pre-decoded pixel image to zoom (avoids repeated decodes)
        """
        if zoom_factor == 1.0:
            return latent

        # Use pre-decoded base image if provided, otherwise decode latent
        if base_pixel_image is not None:
            pixel_samples = base_pixel_image
        else:
            pixel_samples = vae.decode(latent)

        # pixel_samples shape: (B, H, W, C) in ComfyUI format
        # Need to convert to (B, C, H, W) for F.interpolate
        if pixel_samples.shape[-1] == 3:  # (B, H, W, C)
            pixel_samples = pixel_samples.movedim(-1, 1)  # → (B, C, H, W)

        B, C, H, W = pixel_samples.shape

        # Calculate enlarged dimensions
        new_h = int(H * zoom_factor)
        new_w = int(W * zoom_factor)

        # Bicubic enlargement
        enlarged = F.interpolate(
            pixel_samples,
            size=(new_h, new_w),
            mode='bicubic',
            align_corners=False
        )

        # Center crop to original dimensions
        crop_top = (new_h - H) // 2
        crop_left = (new_w - W) // 2
        cropped = enlarged[:, :, crop_top:crop_top + H, crop_left:crop_left + W]

        # Convert back to ComfyUI format (B, H, W, C) for VAE encode
        if C == 3:
            cropped = cropped.movedim(1, -1)  # (B, C, H, W) → (B, H, W, C)

        # VAE encode back to latent space
        re_encoded_latent = vae.encode(cropped)

        return re_encoded_latent

    def process_image(self, model, positive, negative, latent_image, wave_config, seed, add_noise,
                      sampler_name, scheduler, cfg, noise_injection_strength, lock_injection_seed, feedback_mode,
                      vae=None, checkpoint_config=None, prompt=None, extra_pnginfo=None):

        # Validate single latent input
        input_latent = latent_image["samples"]
        if input_latent.shape[0] != 1:
            raise Exception(f"ImageIterativeSampler requires a single latent image (batch size = 1), got batch size = {input_latent.shape[0]}")

        if wave_config is None:
            raise Exception("ImageIterativeSampler requires a wave_config input from WaveController.")

        # Determine number of cycles from CLIP/IPAdapter image sequences, or fall back to cycle_length
        clip_sequence = wave_config.get("clip_vision_sequence") or []
        ipadapter_sequence = wave_config.get("ipadapter_image_sequence") or []

        # If image sequences exist, use their length; otherwise fall back to cycle_length/cycle_count
        if len(clip_sequence) > 0 or len(ipadapter_sequence) > 0:
            num_cycles = max(len(clip_sequence), len(ipadapter_sequence))
            print(f"Cycle count determined by image sequences: {num_cycles}")
        else:
            # Fall back to cycle_count (ImageController) or cycle_length (WaveController)
            num_cycles = max(1, wave_config.get("cycle_count", wave_config.get("cycle_length", 1)))
            print(f"No image sequences found. Using config cycle count: {num_cycles}")

        # Extract wave config parameters
        step_floor = max(1, wave_config.get("step_floor", 5))
        start_at_step = wave_config.get("start_at_step", 0)
        end_at_step = wave_config.get("end_at_step", 20)

        # Check if zoom is enabled (linear interpolation from min to max)
        zoom_rate = wave_config.get("zoom_rate", wave_config.get("zoom_max", 1.0))
        zoom_enabled = zoom_rate != 1.0

        # Validate VAE requirement for zoom
        if zoom_enabled and vae is None:
            raise Exception("ImageIterativeSampler requires a VAE input when zoom is enabled. Please connect a VAE to enable pixel-space zoom processing.")

        # Calculate iterations per cycle
        iterations_per_cycle = (end_at_step - start_at_step) - step_floor + 1
        total_iterations = iterations_per_cycle * num_cycles

        # Extract checkpoint configuration
        checkpoint_config = checkpoint_config or {}
        enable_checkpoint = checkpoint_config.get("checkpoint_enabled", False)
        checkpoint_interval = checkpoint_config.get("checkpoint_interval", 5)
        checkpoint_dir = checkpoint_config.get("checkpoint_dir", "")
        run_id = checkpoint_config.get("checkpoint_run_id", "")
        resume_iteration = checkpoint_config.get("resume_frame", 0)  # Reuse 'resume_frame' key
        loaded_iterations = checkpoint_config.get("loaded_latents", None)
        loaded_cycle_input = checkpoint_config.get("loaded_current_latent", None)
        checkpoint_type = checkpoint_config.get("checkpoint_type", None)

        # Validate checkpoint type if resuming from a checkpoint
        if loaded_iterations is not None and checkpoint_type is not None:
            expected_type = "image_iterative_sampler"
            if checkpoint_type != expected_type:
                raise ValueError(
                    f"Checkpoint type mismatch!\n"
                    f"This sampler expects: '{expected_type}'\n"
                    f"But checkpoint was created by: '{checkpoint_type}'\n"
                    f"Please use the correct sampler for this checkpoint."
                )

        print(f"\n{'='*60}")
        print(f"ImageIterativeSampler: Processing {num_cycles} cycles x {iterations_per_cycle} iterations = {total_iterations} total")
        print(f"  Step progression per cycle: {step_floor} to {end_at_step - start_at_step} steps")
        print(f"  Start at step: {start_at_step}")
        print(f"  End at step: {end_at_step}")
        print(f"  Noise injection: {noise_injection_strength}")
        print(f"  Feedback mode: {feedback_mode}")
        if zoom_enabled:
            # Calculate cumulative zoom after all iterations for display
            cumulative_zoom = zoom_rate ** total_iterations
            print(f"  Zoom: ENABLED (exponential) - {zoom_rate}x per iteration, {cumulative_zoom:.2f}x cumulative after {total_iterations} iterations")
        else:
            print(f"  Zoom: disabled (zoom_rate=1.0)")
        if enable_checkpoint:
            print(f"  Checkpointing: enabled (every {checkpoint_interval} iterations)")
            print(f"  Run ID: {run_id}")
            if resume_iteration > 0:
                print(f"  Resuming from iteration: {resume_iteration}")
            elif vae is not None:
                generate_checkpoint_preview_image(input_latent[0], vae, run_id, checkpoint_dir, prefix="image_ckpt_", prompt=prompt, extra_pnginfo=extra_pnginfo, preview_type="input_latent")
        print(f"{ '='*60}\n")

        # Initialize iteration list (use loaded iterations if resuming)
        processed_iterations = loaded_iterations if loaded_iterations else []

        # Determine which cycle we're in and the cycle input latent
        if resume_iteration > 0:
            resume_cycle = resume_iteration // iterations_per_cycle
            cycle_input_latent = loaded_cycle_input if loaded_cycle_input is not None else input_latent.clone()
            start_iteration = resume_iteration
            # Calculate cycle seed for resume
            cycle_seed = seed + (resume_cycle * 1000)
        else:
            resume_cycle = 0
            cycle_input_latent = input_latent.clone()
            start_iteration = 0
            cycle_seed = seed

        # Extract style/clip/controlnet/ipadapter configuration
        style_model = wave_config.get("style_model")
        strength_type = wave_config.get("style_strength_type", "multiply")
        clip_vision_output = wave_config.get("clip_vision_output")

        ipadapter_enabled = wave_config.get("ipadapter_enabled", False) and IPADAPTER_AVAILABLE
        ipadapter_config = {}
        if ipadapter_enabled:
            ipadapter_model = wave_config.get("ipadapter_model")
            clipvision_model = wave_config.get("ipadapter_clip_vision")

            if isinstance(ipadapter_model, dict) and "ipadapter" in ipadapter_model:
                actual_ipadapter = ipadapter_model["ipadapter"]["model"]
                actual_clipvision = ipadapter_model["clipvision"]["model"]
            else:
                actual_ipadapter = ipadapter_model
                actual_clipvision = clipvision_model

            ipadapter_config = {
                "ipadapter": actual_ipadapter,
                "clipvision": actual_clipvision,
                "image": wave_config.get("ipadapter_image"),
                "weight": wave_config.get("ipadapter_weight", 1.0),
                "weight_type": wave_config.get("ipadapter_weight_type", "linear"),
                "combine_embeds": wave_config.get("ipadapter_combine_embeds", "concat"),
                "start_at": wave_config.get("ipadapter_start_at", 0.0),
                "end_at": wave_config.get("ipadapter_end_at", 1.0),
                "embeds_scaling": wave_config.get("ipadapter_embeds_scaling", "V only"),
                "image_negative": wave_config.get("ipadapter_image_negative"),
                "attn_mask": wave_config.get("ipadapter_attn_mask"),
            }

        controlnet_enabled = wave_config.get("controlnet_model") is not None
        controlnet_config = {}
        if controlnet_enabled:
            controlnet_config = {
                "control_net": wave_config.get("controlnet_model"),
                "control_hint": wave_config.get("controlnet_hint"),
                "vae": wave_config.get("controlnet_vae"),
                "strength": 1.0,
                "start_at": 0.0,
                "end_at": 1.0,
            }

        # Decode base image once per cycle (if zoom enabled)
        base_pixel_image = None
        if zoom_enabled:
            base_pixel_image = vae.decode(cycle_input_latent)

        # Cache style conditioning once per cycle (if style model enabled)
        cycle_style_cond = None
        if style_model is not None and clip_vision_output is not None:
            cycle_style_cond = style_model.get_cond(clip_vision_output).flatten(start_dim=0, end_dim=1).unsqueeze(dim=0)

        # Cache noise latent once per cycle when seed is locked (avoids repeated VAE encodes)
        cycle_noise_latent = None
        if lock_injection_seed and noise_injection_strength > 0 and vae is not None:
            noise_seed = seed + 1  # Initial cycle seed offset
            cycle_noise_latent = generate_noise_latent(input_latent.shape, vae, noise_seed)

        # Main iteration loop - process cycles
        for iteration_idx in range(start_iteration, total_iterations):
            # Determine which cycle we're in and position within cycle
            cycle_idx = iteration_idx // iterations_per_cycle
            iteration_in_cycle = iteration_idx % iterations_per_cycle

            # Check if we're starting a new cycle
            if iteration_in_cycle == 0 and iteration_idx > 0:
                # New cycle: use final output from previous cycle as input
                cycle_input_latent = processed_iterations[-1].clone()
                # Decode new base image for this cycle (if zoom enabled)
                if zoom_enabled:
                    base_pixel_image = vae.decode(cycle_input_latent)
                # Generate new cycle seed to prevent burn-in artifacts
                cycle_seed = seed + (cycle_idx * 1000)
                # Regenerate cached noise latent for new cycle (if seed is locked)
                if lock_injection_seed and noise_injection_strength > 0 and vae is not None:
                    noise_seed = cycle_seed + 1
                    cycle_noise_latent = generate_noise_latent(cycle_input_latent.shape, vae, noise_seed)
                print(f"\n--- Starting Cycle {cycle_idx + 1}/{num_cycles} (seed: {cycle_seed}) ---")

            # Calculate progressive steps for this iteration within the cycle
            current_steps = step_floor + iteration_in_cycle
            current_end = start_at_step + current_steps
            actual_end = min(current_end, end_at_step)
            actual_steps = actual_end - start_at_step

            # Calculate zoom and clip strength for this iteration
            # Exponential zoom naturally chains across cycles without needing relative calculation
            zoom, clip_strength = self.calculate_wave_values(iteration_in_cycle, wave_config, iterations_per_cycle)

            # Per-iteration seed for noise injection
            if lock_injection_seed and noise_injection_strength > 0:
                noise_seed = cycle_seed + 1
            else:
                noise_seed = cycle_seed + iteration_in_cycle
            # Main sampler seed changes per cycle to prevent burn-in
            sampler_seed = cycle_seed

            # Select CLIP image for this cycle
            iteration_positive = positive
            iteration_clip_output = clip_vision_output
            if clip_sequence:
                iteration_clip_output = clip_sequence[cycle_idx % len(clip_sequence)]

            # Recompute style conditioning when starting new cycle (CLIP may have changed)
            if iteration_in_cycle == 0 and style_model is not None and iteration_clip_output is not None:
                cycle_style_cond = style_model.get_cond(iteration_clip_output).flatten(start_dim=0, end_dim=1).unsqueeze(dim=0)

            if style_model is not None and iteration_clip_output is not None:
                iteration_positive = apply_style_model_wrapper(positive, style_model, iteration_clip_output,
                                                            clip_strength, strength_type, base_cond=cycle_style_cond)

            # Apply ControlNet
            iteration_negative = negative
            if controlnet_enabled:
                cn_strength = controlnet_config["strength"]
                cn_start_at = controlnet_config["start_at"]
                cn_end_at = controlnet_config["end_at"]

                if "controlnet_strength_min" in wave_config:
                    cn_strength, cn_start_at, cn_end_at = self.calculate_controlnet_wave_values(
                        iteration_idx, wave_config, total_iterations
                    )

                control_hint_batch = controlnet_config["control_hint"]
                batch_size = control_hint_batch.shape[0]
                hint_idx = cycle_idx % batch_size
                iteration_control_hint = control_hint_batch[hint_idx:hint_idx+1]

                iteration_positive, iteration_negative = apply_controlnet_wrapper(
                    iteration_positive, iteration_negative,
                    controlnet_config["control_net"],
                    iteration_control_hint,
                    cn_strength, cn_start_at, cn_end_at,
                    vae=controlnet_config["vae"]
                )

            # Apply IPAdapter
            iteration_model = model
            if ipadapter_enabled:
                ipadapter_weight = ipadapter_config["weight"]
                ipadapter_start_at = ipadapter_config["start_at"]
                ipadapter_end_at = ipadapter_config["end_at"]

                if "ipadapter_weight_min" in wave_config:
                    ipadapter_weight, ipadapter_start_at, ipadapter_end_at = self.calculate_ipadapter_wave_values(
                        iteration_idx, wave_config, total_iterations
                    )

                iteration_image = ipadapter_config["image"]
                if ipadapter_sequence:
                    iteration_image = ipadapter_sequence[cycle_idx % len(ipadapter_sequence)]

                iteration_model = model.clone()
                iteration_model, _ = ipadapter_execute(
                    iteration_model,
                    ipadapter_config["ipadapter"],
                    ipadapter_config["clipvision"],
                    image=iteration_image,
                    weight=ipadapter_weight,
                    weight_type=ipadapter_config["weight_type"],
                    combine_embeds=ipadapter_config["combine_embeds"],
                    start_at=ipadapter_start_at,
                    end_at=ipadapter_end_at,
                    embeds_scaling=ipadapter_config["embeds_scaling"],
                    image_negative=ipadapter_config["image_negative"],
                    attn_mask=ipadapter_config["attn_mask"],
                )

            print(f"Cycle {cycle_idx+1}/{num_cycles}, Iter {iteration_in_cycle+1}/{iterations_per_cycle} (total {iteration_idx+1}/{total_iterations}): steps={actual_steps}, start_step={start_at_step}, end_step={actual_end}, zoom={zoom:.3f}, clip_strength={clip_strength:.3f}")

            # Apply zoom to base image BEFORE denoising (if enabled)
            if zoom_enabled and zoom != 1.0:
                current_latent = self.zoom_latent_pixel_space(cycle_input_latent, zoom, vae, base_pixel_image=base_pixel_image)
            else:
                current_latent = cycle_input_latent.clone()

            # Apply feedback mode
            if feedback_mode == "previous_iteration" and len(processed_iterations) > 0:
                alpha = 0.1
                current_latent = current_latent * (1 - alpha) + processed_iterations[-1] * alpha

            # Apply noise injection (zoom moved to after denoising)
            if noise_injection_strength > 0:
                current_latent = inject_noise_latent(current_latent, noise_injection_strength, vae=vae, seed=noise_seed, cached_noise_latent=cycle_noise_latent)

            latent_dict = {"samples": current_latent}

            # Sample with progressive steps
            result = nodes.common_ksampler(
                iteration_model, sampler_seed, actual_end, cfg, sampler_name, scheduler,
                iteration_positive, iteration_negative, latent_dict,
                disable_noise=(not add_noise), start_step=start_at_step, last_step=10000, force_full_denoise=True
            )

            processed_latent = result[0]["samples"]

            processed_iterations.append(processed_latent)

            # Save checkpoint at intervals (save current cycle_input_latent for resume)
            if enable_checkpoint and (iteration_idx + 1) % checkpoint_interval == 0:
                stacked = torch.cat(processed_iterations, dim=0)
                tensors = {
                    "latent_tensor": stacked.contiguous(),
                    "current_latent": cycle_input_latent.contiguous()
                }
                metadata = {
                    "iteration_count": str(len(processed_iterations)),
                    "last_iteration_idx": str(iteration_idx),
                    "total_iterations": str(total_iterations),
                    "checkpoint_type": "image_iterative_sampler"
                }
                checkpoint_path = save_checkpoint_file(tensors, metadata, checkpoint_dir, run_id, prefix="image_ckpt_")
                
                if checkpoint_path:
                    print(f"✓ Checkpoint saved at iteration {iteration_idx + 1}/{total_iterations}")

        all_iterations = torch.cat(processed_iterations, dim=0)

        print(f"\n{'='*60}")
        print(f"ImageIterativeSampler: Complete! Processed {num_cycles} cycles x {iterations_per_cycle} iterations = {total_iterations} total")

        # Cleanup checkpoint on successful completion
        if enable_checkpoint:
            cleanup_checkpoint_files(checkpoint_dir, run_id, prefix="image_ckpt_")

        print(f"  Clearing VRAM cache...")
        print(f"{ '='*60}\n")

        # Light VRAM cleanup (don't unload all models globally)
        comfy.model_management.soft_empty_cache()

        return ({"samples": all_iterations},)


NODE_CLASS_MAPPINGS = {
    "ImageIterativeSampler": ImageIterativeSampler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageIterativeSampler": "Image Iterative Sampler"
}