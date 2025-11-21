import torch
import torch.nn.functional as F
import comfy.samplers
import comfy.utils
import comfy.model_management
import nodes
import random
import numpy as np
from ..utils.kentskooking_utils import (
    triangle_wave,
    calculate_wave,
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


class VideoIterativeSampler:
    """
    Iterative sampler variant with placeholders for start/end step metadata.
    Currently mirrors the baseline sampler while exposing renamed inputs so
    future KSamplerAdvanced integration can hook into the wave configuration.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_batch": ("LATENT",),
                "wave_config": ("WAVE_CONFIG",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "add_noise": ("BOOLEAN", {"default": True, "label_on": "enable", "label_off": "disable"}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "noise_injection_strength": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "lock_injection_seed": ("BOOLEAN", {"default": False, "label_on": "locked", "label_off": "varying"}),
                "feedback_mode": (["none", "previous_frame"], {"default": "none"}),
            },
            "optional": {
                "vae": ("VAE",),
                "checkpoint_config": ("CHECKPOINT_CONFIG",),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent_batch",)
    FUNCTION = "process_video"
    CATEGORY = "kentskooking/sampling"

    def calculate_wave_values(self, frame_idx, config):
        """
        Calculate advanced steps/zoom/clip for the current frame.
        Always assumes WaveController configuration.
        """
        wave_type = config.get("wave_type", "triangle")
        position_in_cycle = frame_idx % config["cycle_length"]

        step_floor = max(1, config["step_floor"])
        start_at = config["start_at_step"]
        min_end = start_at + step_floor
        max_end = max(min_end, config["end_at_step"])
        current_end = calculate_wave(wave_type, position_in_cycle, config["cycle_length"], min_end, max_end)
        end_at = int(round(max(current_end, min_end)))
        steps = max(step_floor, end_at - start_at)

        zoom = calculate_wave(wave_type, position_in_cycle, config["cycle_length"],
                             config["zoom_min"], config["zoom_max"])
        clip_strength = calculate_wave(wave_type, position_in_cycle, config["cycle_length"],
                                      config["clip_strength_min"], config["clip_strength_max"])

        return steps, zoom, clip_strength, start_at, end_at

    def calculate_ipadapter_wave_values(self, frame_idx, config):
        """
        Calculate IPAdapter wave values for current frame.
        """
        wave_type = config.get("wave_type", "triangle")
        position_in_cycle = frame_idx % config["cycle_length"]

        weight = calculate_wave(wave_type, position_in_cycle, config["cycle_length"],
                              config["ipadapter_weight_min"], config["ipadapter_weight_max"])
        start_at = calculate_wave(wave_type, position_in_cycle, config["cycle_length"],
                                config["ipadapter_start_at_min"], config["ipadapter_start_at_max"])
        end_at = calculate_wave(wave_type, position_in_cycle, config["cycle_length"],
                              config["ipadapter_end_at_min"], config["ipadapter_end_at_max"])

        return weight, start_at, end_at

    def calculate_controlnet_wave_values(self, frame_idx, config):
        """
        Calculate ControlNet wave values for current frame.
        """
        wave_type = config.get("wave_type", "triangle")
        position_in_cycle = frame_idx % config["cycle_length"]

        strength = calculate_wave(wave_type, position_in_cycle, config["cycle_length"],
                                config["controlnet_strength_min"], config["controlnet_strength_max"])
        start_at = calculate_wave(wave_type, position_in_cycle, config["cycle_length"],
                                config["controlnet_start_at_min"], config["controlnet_start_at_max"])
        end_at = calculate_wave(wave_type, position_in_cycle, config["cycle_length"],
                              config["controlnet_end_at_min"], config["controlnet_end_at_max"])

        return strength, start_at, end_at

    def zoom_latent(self, latent, zoom_factor):
        if zoom_factor == 1.0:
            return latent
        batch, channels, height, width = latent.shape
        if zoom_factor > 1.0:
            new_height = int(height / zoom_factor)
            new_width = int(width / zoom_factor)
            top = (height - new_height) // 2
            left = (width - new_width) // 2
            cropped = latent[:, :, top:top+new_height, left:left+new_width]
            zoomed = F.interpolate(cropped, size=(height, width), mode='bicubic', align_corners=False)
        else:
            new_height = int(height * zoom_factor)
            new_width = int(width * zoom_factor)
            scaled = F.interpolate(latent, size=(new_height, new_width), mode='bicubic', align_corners=False)
            pad_top = (height - new_height) // 2
            pad_left = (width - new_width) // 2
            pad_bottom = height - new_height - pad_top
            pad_right = width - new_width - pad_left
            zoomed = F.pad(scaled, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
        return zoomed

    def process_video(self, model, positive, negative, latent_batch, wave_config, seed, add_noise,
                      sampler_name, scheduler, cfg, noise_injection_strength, lock_injection_seed, feedback_mode,
                      vae=None, checkpoint_config=None, prompt=None, extra_pnginfo=None):
        latents = latent_batch["samples"]
        num_frames = latents.shape[0]

        if wave_config is None:
            raise Exception("VideoIterativeSampler requires a wave_config input from WaveController.")

        # Extract checkpoint configuration
        checkpoint_config = checkpoint_config or {}
        enable_checkpoint = checkpoint_config.get("checkpoint_enabled", False)
        checkpoint_interval = checkpoint_config.get("checkpoint_interval", 16)
        checkpoint_dir = checkpoint_config.get("checkpoint_dir", "")
        run_id = checkpoint_config.get("checkpoint_run_id", "")
        resume_frame = checkpoint_config.get("resume_frame", 0)
        loaded_latents = checkpoint_config.get("loaded_latents", None)
        checkpoint_type = checkpoint_config.get("checkpoint_type", None)

        # Validate checkpoint type if resuming from a checkpoint
        if loaded_latents is not None and checkpoint_type is not None:
            expected_type = "video_iterative_sampler"
            if checkpoint_type != expected_type:
                raise ValueError(
                    f"Checkpoint type mismatch!\n"
                    f"This sampler expects: '{expected_type}'\n"
                    f"But checkpoint was created by: '{checkpoint_type}'\n"
                    f"Please use the correct sampler for this checkpoint."
                )

        print(f"\n{'='*60}")
        print(f"VideoIterativeSamplerAdvanced: Processing {num_frames} frames")
        print(f"  Noise injection: {noise_injection_strength}")
        print(f"  Feedback mode: {feedback_mode}")
        print(f"  Wave cycle length: {wave_config.get('cycle_length', 'unknown')}")
        if enable_checkpoint:
            print(f"  Checkpointing: enabled (every {checkpoint_interval} frames)")
            print(f"  Run ID: {run_id}")
            if resume_frame > 0:
                print(f"  Resuming from frame: {resume_frame}")
            elif vae is not None:
                # Generate preview PNG for new checkpointed runs (with workflow metadata for drag-and-drop restore)
                generate_checkpoint_preview_image(latents[0], vae, run_id, checkpoint_dir, prompt=prompt, extra_pnginfo=extra_pnginfo, preview_type="input_latent_first_frame")
        print(f"{'='*60}\n")

        # Initialize latent list (use loaded latents if resuming)
        processed_latents = loaded_latents if loaded_latents else []
        previous_latent = processed_latents[-1].clone() if processed_latents else None
        start_frame = resume_frame

        style_model = None
        strength_type = "multiply"
        clip_vision_output = None
        clip_sequence = []
        ipadapter_enabled = False
        ipadapter_config = {}
        controlnet_enabled = False
        controlnet_config = {}

        style_model = wave_config.get("style_model")
        strength_type = wave_config.get("style_strength_type", "multiply")
        clip_vision_output = wave_config.get("clip_vision_output")
        clip_sequence = wave_config.get("clip_vision_sequence") or []

        # Check for IPAdapter configuration
        ipadapter_enabled = wave_config.get("ipadapter_enabled", False) and IPADAPTER_AVAILABLE
        if ipadapter_enabled:
            # Extract ipadapter from wave_config
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

        # Check for ControlNet configuration
        controlnet_enabled = wave_config.get("controlnet_model") is not None
        if controlnet_enabled:
            controlnet_config = {
                "control_net": wave_config.get("controlnet_model"),
                "control_hint": wave_config.get("controlnet_hint"),
                "vae": wave_config.get("controlnet_vae"),
                "strength": 1.0,  # default, will be overridden by wave if present
                "start_at": 0.0,
                "end_at": 1.0,
            }

        # Cache noise latent when seed is locked (avoids repeated VAE encodes for all frames)
        cached_noise_latent = None
        if lock_injection_seed and noise_injection_strength > 0 and vae is not None:
            noise_seed = seed + 1  # Locked seed
            cached_noise_latent = generate_noise_latent(latents[0:1].shape, vae, noise_seed)

        for frame_idx in range(start_frame, num_frames):
            current_latent = latents[frame_idx:frame_idx+1]

            steps, zoom, clip_strength, start_step, end_step = self.calculate_wave_values(frame_idx, wave_config)

            # Per-frame seed for noise injection variation (small randomness)
            # If lock_injection_seed is enabled and noise injection is active, use constant seed
            if lock_injection_seed and noise_injection_strength > 0:
                noise_seed = seed + 1  # Locked seed (different from sampler seed, but constant across frames)
            else:
                noise_seed = seed + frame_idx  # Varying seed per frame
            # Main sampler seed stays constant for video consistency
            sampler_seed = seed

            frame_positive = positive
            frame_clip_output = clip_vision_output
            if clip_sequence:
                seq_len = len(clip_sequence)
                cycle_len = max(1, wave_config.get("cycle_length", 1))
                cycle_index = (frame_idx // cycle_len) % seq_len
                frame_clip_output = clip_sequence[cycle_index]

            if style_model is not None and frame_clip_output is not None:
                frame_positive = apply_style_model_wrapper(positive, style_model, frame_clip_output,
                                                        clip_strength, strength_type)

            # Apply ControlNet per-frame with dynamic wave parameters
            frame_negative = negative
            if controlnet_enabled:
                # Calculate dynamic ControlNet parameters from wave if available
                cn_strength = controlnet_config["strength"]
                cn_start_at = controlnet_config["start_at"]
                cn_end_at = controlnet_config["end_at"]

                # Check if wave parameters are present for ControlNet modulation
                if "controlnet_strength_min" in wave_config:
                    cn_strength, cn_start_at, cn_end_at = self.calculate_controlnet_wave_values(
                        frame_idx, wave_config
                    )

                # Extract the correct frame from control_hint batch
                control_hint_batch = controlnet_config["control_hint"]
                batch_size = control_hint_batch.shape[0]
                hint_idx = frame_idx % batch_size
                frame_control_hint = control_hint_batch[hint_idx:hint_idx+1]

                # Apply controlnet to both positive and negative conditioning
                frame_positive, frame_negative = apply_controlnet_wrapper(
                    frame_positive, frame_negative,
                    controlnet_config["control_net"],
                    frame_control_hint,
                    cn_strength, cn_start_at, cn_end_at,
                    vae=controlnet_config["vae"]
                )

            # Apply IPAdapter per-frame with dynamic wave parameters
            frame_model = model
            if ipadapter_enabled:
                # Calculate dynamic IPAdapter parameters from wave if available
                ipadapter_weight = ipadapter_config["weight"]
                ipadapter_start_at = ipadapter_config["start_at"]
                ipadapter_end_at = ipadapter_config["end_at"]

                # Check if wave parameters are present for IPAdapter modulation
                if "ipadapter_weight_min" in wave_config:
                    ipadapter_weight, ipadapter_start_at, ipadapter_end_at = self.calculate_ipadapter_wave_values(
                        frame_idx, wave_config
                    )

                # Select image from sequence based on cycle
                frame_image = ipadapter_config["image"]
                image_sequence = wave_config.get("ipadapter_image_sequence", [])
                if image_sequence:
                    seq_len = len(image_sequence)
                    cycle_len = max(1, wave_config.get("cycle_length", 1))
                    cycle_index = (frame_idx // cycle_len) % seq_len
                    frame_image = image_sequence[cycle_index]

                # Clone model and apply IPAdapter with calculated values
                frame_model = model.clone()
                frame_model, _ = ipadapter_execute(
                    frame_model,
                    ipadapter_config["ipadapter"],
                    ipadapter_config["clipvision"],
                    image=frame_image,
                    weight=ipadapter_weight,
                    weight_type=ipadapter_config["weight_type"],
                    combine_embeds=ipadapter_config["combine_embeds"],
                    start_at=ipadapter_start_at,
                    end_at=ipadapter_end_at,
                    embeds_scaling=ipadapter_config["embeds_scaling"],
                    image_negative=ipadapter_config["image_negative"],
                    attn_mask=ipadapter_config["attn_mask"],
                )

            print(f"Frame {frame_idx+1}/{num_frames}: steps={steps}, start_step={start_step}, end_step={end_step}, zoom={zoom:.3f}, clip_strength={clip_strength:.3f}")

            if feedback_mode == "previous_frame" and previous_latent is not None:
                alpha = 0.1
                current_latent = current_latent * (1 - alpha) + previous_latent * alpha

            if zoom != 1.0:
                current_latent = self.zoom_latent(current_latent, zoom)

            if noise_injection_strength > 0:
                current_latent = inject_noise_latent(current_latent, noise_injection_strength, vae=vae, seed=noise_seed, cached_noise_latent=cached_noise_latent)

            latent_dict = {"samples": current_latent}

            # Advanced sampling approach (matching native KSamplerAdvanced):
            # - end_step defines the total sigma schedule length
            # - start_step is where we begin in that schedule
            # - last_step=10000 is a fallback (effectively ignored since it's > end_step)
            # - sampler_seed stays constant across all frames for video consistency
            # Example: start=20, end=40 runs steps 20-40 of a 40-step schedule (20 actual steps)
            result = nodes.common_ksampler(
                frame_model, sampler_seed, end_step, cfg, sampler_name, scheduler,
                frame_positive, frame_negative, latent_dict,
                disable_noise=(not add_noise), start_step=start_step, last_step=10000, force_full_denoise=True
            )

            processed_latent = result[0]["samples"]
            processed_latents.append(processed_latent)
            previous_latent = processed_latent.clone()

            # Save checkpoint at intervals
            if enable_checkpoint and (frame_idx + 1) % checkpoint_interval == 0:
                stacked = torch.cat(processed_latents, dim=0)
                tensors = {"latent_tensor": stacked.contiguous()}
                metadata = {
                   "frame_count": str(len(processed_latents)),
                   "last_frame_idx": str(frame_idx),
                   "checkpoint_type": "video_iterative_sampler"
                }
                checkpoint_path = save_checkpoint_file(tensors, metadata, checkpoint_dir, run_id)
                
                if checkpoint_path:
                    print(f"✓ Checkpoint saved at frame {frame_idx + 1}/{num_frames}")

        all_latents = torch.cat(processed_latents, dim=0)

        print(f"\n{'='*60}")
        print(f"VideoIterativeSamplerAdvanced: Complete! Processed {num_frames} frames")

        # Cleanup checkpoint on successful completion
        if enable_checkpoint:
            cleanup_checkpoint_files(checkpoint_dir, run_id)

        print(f"  Clearing VRAM cache...")
        print(f"{'='*60}\n")

        # Light VRAM cleanup (don't unload all models globally)
        comfy.model_management.soft_empty_cache()

        return ({"samples": all_latents},)


NODE_CLASS_MAPPINGS = {
    "VideoIterativeSampler": VideoIterativeSampler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoIterativeSampler": "Video Iterative Sampler"
}