import torch
import nodes
import comfy.samplers
import comfy.utils
import comfy.model_management
from ..utils.kentskooking_utils import calculate_wave, apply_controlnet_wrapper, apply_style_model_wrapper

# Import IPAdapter classes
try:
    import custom_nodes.ComfyUI_IPAdapter_plus.IPAdapterPlus as IPAdapterModule
    # We only need IPAdapterTiled as it handles the tiling logic and calls ipadapter_execute internally
    IPAdapterTiled = IPAdapterModule.IPAdapterTiled
    IPADAPTER_AVAILABLE = True
except ImportError:
    IPADAPTER_AVAILABLE = False

class VideoInterpolateSampler:
    """
    Specialized sampler for morphing between two video sequences.
    Concatenates Video A + Morph(A_end + B_start) + Video B.
    Applies bell-curve denoising only during the morph zone.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive_A": ("CONDITIONING",),
                "positive_B": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latents_A": ("LATENT",),
                "latents_B": ("LATENT",),
                "wave_config": ("WAVE_CONFIG",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "feedback_mode": (["none", "previous_frame"], {"default": "none"}),
            },
            "optional": {
                "vae": ("VAE",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent_batch",)
    FUNCTION = "process_interpolation"
    CATEGORY = "kentskooking/sampling"

    def interpolate_conditioning(self, cond_A, cond_B, t):
        """
        Interpolate between two conditioning objects.
        Since conditioning is complex (list of tensors/dicts), we simply return
        weighted average of the embeddings.
        
        For the dictionary part (which holds ControlNets, GLIGEN, etc.), we cannot
        mathematically 'blend' the objects easily. 
        Strategy: Switch priority based on 't'.
        - t < 0.5: Use dict_A as base, update with NON-colliding keys from B.
        - t >= 0.5: Use dict_B as base, update with NON-colliding keys from A.
        """
        
        out_cond = []
        # We assume structure matches (same number of conditioning chunks)
        for i in range(len(cond_A)):
            c_A = cond_A[i][0]
            c_B = cond_B[i][0] if i < len(cond_B) else c_A # Fallback
            
            if c_A.shape != c_B.shape:
                 raise ValueError(f"Conditioning shape mismatch at index {i}: A={c_A.shape}, B={c_B.shape}. "
                                  "Cannot interpolate between different text encoder models or prompts with vastly different token counts if not padded.")

            # Linear interpolation of embeddings: A * (1-t) + B * t
            blended_c = c_A * (1.0 - t) + c_B * t
            
            # Intelligent Dictionary Merging
            dict_A = cond_A[i][1]
            dict_B = cond_B[i][1] if i < len(cond_B) else dict_A
            
            if t < 0.5:
                # Closer to A: Use A's controlnets/attributes
                final_dict = dict_A.copy()
            else:
                # Closer to B
                final_dict = dict_B.copy()
                
            # Remove our custom key from the blended result to prevent double-application or confusion
            # We will re-apply them explicitly in the main loop
            if "kentskooking_controlnet" in final_dict:
                del final_dict["kentskooking_controlnet"]
            
            # CRITICAL: Remove any existing 'control' object to prevent double-application
            # The loop below will dynamically apply the correct ControlNet frame and strength.
            if "control" in final_dict:
                del final_dict["control"]

            out_cond.append([blended_c, final_dict])
            
        return out_cond

    def process_interpolation(self, model, positive_A, negative, latents_A, latents_B, wave_config, seed, steps, sampler_name, scheduler, cfg, feedback_mode, positive_B=None, vae=None):
        
        # Handle optional positive_B
        if positive_B is None:
            positive_B = positive_A

        tensor_A = latents_A["samples"]
        tensor_B = latents_B["samples"]
        
        # Validation: Check dimensions (Batch, Channel, Height, Width)
        if tensor_A.shape[1:] != tensor_B.shape[1:]:
            raise ValueError(f"Latent Shape Mismatch! Video A: {tensor_A.shape}, Video B: {tensor_B.shape}. "
                             "Videos must have the same height, width, and channel count to be interpolated.")

        len_A = tensor_A.shape[0]
        len_B = tensor_B.shape[0]
        
        overlap_frames = wave_config.get("overlap_frames", 25)
        
        if len_A < overlap_frames or len_B < overlap_frames:
            raise ValueError(f"Video inputs too short! Overlap is {overlap_frames}, but videos are {len_A} and {len_B} frames.")

        # 1. Slice Segments
        seg1_A = tensor_A[:-overlap_frames]
        morph_A = tensor_A[-overlap_frames:]
        morph_B = tensor_B[:overlap_frames]
        seg3_B = tensor_B[overlap_frames:]

        if seg1_A.shape[0] == 0 and seg3_B.shape[0] == 0:
            print(f"\n⚠️  WARNING: Overlap ({overlap_frames}) equals video length - entire output will be morphed!")

        print(f"\n{'='*60}")
        print(f"VideoInterpolateSampler: Stitching Seamless Video")
        print(f"  Part 1 (Source A): {seg1_A.shape[0]} frames")
        print(f"  Part 2 (Morph):    {overlap_frames} frames")
        print(f"  Part 3 (Source B): {seg3_B.shape[0]} frames")
        print(f"  Total Output:      {seg1_A.shape[0] + overlap_frames + seg3_B.shape[0]} frames")
        print(f"{'='*60}\n")

        # --------------------------------------------------------------------------------
        # Setup & Config Extraction
        # --------------------------------------------------------------------------------
        ipadapter_enabled = wave_config.get("ipadapter_enabled", False) and IPADAPTER_AVAILABLE

        # Initialize Tiled Processor if needed
        tiled_processor = None
        if ipadapter_enabled:
            tiled_processor = IPAdapterTiled()
            # Enable batch unfolding for weight lists
            tiled_processor.unfold_batch = True 

        # Extract IPAdapter Data
        ipadapter_data = {}
        if ipadapter_enabled:
            # ... (Existing extraction logic) ...
            ipadapter_model = wave_config.get("ipadapter_model")
            clipvision_model = wave_config.get("ipadapter_clip_vision")
            
            if isinstance(ipadapter_model, dict) and "ipadapter" in ipadapter_model:
                actual_ipadapter = ipadapter_model["ipadapter"]["model"]
                actual_clipvision = ipadapter_model["clipvision"]["model"]
            else:
                actual_ipadapter = ipadapter_model
                actual_clipvision = clipvision_model
                
            ipadapter_data = {
                "ipadapter": actual_ipadapter,
                "clipvision": actual_clipvision,
                "weight": wave_config.get("ipadapter_weight", 1.0),
                "weight_type": wave_config.get("ipadapter_weight_type", "linear"),
                "combine_embeds": wave_config.get("ipadapter_combine_embeds", "concat"),
                "start_at": wave_config.get("ipadapter_start_at", 0.0),
                "end_at": wave_config.get("ipadapter_end_at", 1.0),
                "embeds_scaling": wave_config.get("ipadapter_embeds_scaling", "V only"),
                "image_negative": wave_config.get("ipadapter_image_negative"),
                "attn_mask": wave_config.get("ipadapter_attn_mask"),
            }
            
            # Image Sequence Logic
            seq = wave_config.get("ipadapter_image_sequence") or []
            single_img = wave_config.get("ipadapter_image")
            frames_A = []
            frames_B = []
            if len(seq) >= 2 * overlap_frames:
                frames_A = seq[:overlap_frames]
                frames_B = seq[overlap_frames:2*overlap_frames]
            elif len(seq) >= 2:
                frames_A = [seq[0]] * overlap_frames
                frames_B = [seq[1]] * overlap_frames
            elif single_img is not None:
                frames_A = [single_img] * overlap_frames
                frames_B = [single_img] * overlap_frames
            else:
                ipadapter_enabled = False
            
            ipadapter_data["frames_A"] = frames_A
            ipadapter_data["frames_B"] = frames_B

        # --------------------------------------------------------------------------------
        # Style Model Setup
        # --------------------------------------------------------------------------------
        style_model = wave_config.get("style_model")
        style_strength_type = wave_config.get("style_strength_type", "multiply")
        style_frames_A = []
        style_frames_B = []
        
        if style_model is not None:
            style_seq = wave_config.get("clip_vision_sequence") or []
            style_single = wave_config.get("clip_vision_output")
            
            if len(style_seq) >= 2 * overlap_frames:
                # Sequence A (first half), Sequence B (second half)
                style_frames_A = style_seq[:overlap_frames]
                style_frames_B = style_seq[overlap_frames:2*overlap_frames]
            elif len(style_seq) >= 2:
                # Static A, Static B
                style_frames_A = [style_seq[0]] * overlap_frames
                style_frames_B = [style_seq[1]] * overlap_frames
            elif style_single is not None:
                # Single style for both
                style_frames_A = [style_single] * overlap_frames
                style_frames_B = [style_single] * overlap_frames
            else:
                style_model = None # Disable if no clips

        # --------------------------------------------------------------------------------
        # ITERATIVE MODE (Frame-by-Frame)
        # --------------------------------------------------------------------------------
        
        processed_morph_frames_list = []

        # Initialize previous latent
        if seg1_A.shape[0] > 0:
            previous_latent = seg1_A[-1:]
        else:
            previous_latent = morph_A[0:1]

        for i in range(overlap_frames):
            t = i / (overlap_frames - 1) if overlap_frames > 1 else 0.5
            
            # Blend Latents
            latent_a_frame = morph_A[i:i+1]
            latent_b_frame = morph_B[i:i+1]
            blended_latent = latent_a_frame * (1.0 - t) + latent_b_frame * t
            
            if feedback_mode == "previous_frame":
                alpha = 0.1
                blended_latent = blended_latent * (1.0 - alpha) + previous_latent * alpha
            
            # Blend Conditioning
            current_positive = self.interpolate_conditioning(positive_A, positive_B, t)
            current_negative = negative 
            
            # Apply Style Model (Iterative A -> B)
            if style_model is not None:
                # Calculate modulated strength using wave settings if available
                clip_min = wave_config.get("clip_strength_min", 1.0)
                clip_max = wave_config.get("clip_strength_max", 1.0)
                wave_type = wave_config.get("wave_type", "triangle")
                
                # Base strength for this frame (usually uniform max, or wave modulated)
                # Since we are morphing, we usually want 'max' strength overall, split between A and B.
                # But we can allow the 'overall' intensity to dip/peak based on the wave controller.
                overall_strength = calculate_wave(wave_type, i, overlap_frames, clip_min, clip_max)
                
                # Strength A: Fades Out
                strength_A = overall_strength * (1.0 - t)
                # Strength B: Fades In
                strength_B = overall_strength * t
                
                # Apply A
                if strength_A > 0:
                    current_positive = apply_style_model_wrapper(
                        current_positive, style_model, style_frames_A[i],
                        strength_A, style_strength_type
                    )
                
                # Apply B
                if strength_B > 0:
                    current_positive = apply_style_model_wrapper(
                        current_positive, style_model, style_frames_B[i],
                        strength_B, style_strength_type
                    )

            # Apply ControlNets (Iterative)
            if len(positive_A) > 0 and "kentskooking_controlnet" in positive_A[0][1]:
                 # ... (CN Logic A) ...
                 cn_data = positive_A[0][1]["kentskooking_controlnet"]
                 frame_idx = len_A - overlap_frames + i
                 batch_size = cn_data["control_hint"].shape[0]
                 hint_idx = frame_idx % batch_size
                 hint = cn_data["control_hint"][hint_idx:hint_idx+1]
                 w_conf = cn_data["wave_config"]
                 s_min = w_conf.get("controlnet_strength_min", 0.0)
                 s_max = w_conf.get("controlnet_strength_max", 1.0)
                 start_min = w_conf.get("controlnet_start_at_min", 0.0)
                 start_max = w_conf.get("controlnet_start_at_max", 0.0)
                 end_min = w_conf.get("controlnet_end_at_min", 1.0)
                 end_max = w_conf.get("controlnet_end_at_max", 1.0)
                 strength = s_max * (1.0 - t) + s_min * t
                 start_at = start_max * (1.0 - t) + start_min * t
                 end_at = end_max * (1.0 - t) + end_min * t
                 current_positive, current_negative = apply_controlnet_wrapper(
                     current_positive, current_negative,
                     cn_data["control_net"], hint, strength, start_at, end_at, vae=cn_data["vae"]
                 )

            if len(positive_B) > 0 and "kentskooking_controlnet" in positive_B[0][1]:
                 # ... (CN Logic B) ...
                 cn_data = positive_B[0][1]["kentskooking_controlnet"]
                 frame_idx = i
                 batch_size = cn_data["control_hint"].shape[0]
                 hint_idx = frame_idx % batch_size
                 hint = cn_data["control_hint"][hint_idx:hint_idx+1]
                 w_conf = cn_data["wave_config"]
                 s_min = w_conf.get("controlnet_strength_min", 0.0)
                 s_max = w_conf.get("controlnet_strength_max", 1.0)
                 start_min = w_conf.get("controlnet_start_at_min", 0.0)
                 start_max = w_conf.get("controlnet_start_at_max", 0.0)
                 end_min = w_conf.get("controlnet_end_at_min", 1.0)
                 end_max = w_conf.get("controlnet_end_at_max", 1.0)
                 strength = s_min * (1.0 - t) + s_max * t
                 start_at = start_min * (1.0 - t) + start_max * t
                 end_at = end_min * (1.0 - t) + end_max * t
                 current_positive, current_negative = apply_controlnet_wrapper(
                     current_positive, current_negative,
                     cn_data["control_net"], hint, strength, start_at, end_at, vae=cn_data["vae"]
                 )

            # Apply IPAdapter (Iterative)
            current_model = model
            if ipadapter_enabled:
                w_static = ipadapter_data["weight"]
                w_min = wave_config.get("ipadapter_weight_min", 0.0)
                w_max = wave_config.get("ipadapter_weight_max", w_static)
                start_static = ipadapter_data["start_at"]
                start_min = wave_config.get("ipadapter_start_at_min", 0.0)
                start_max = wave_config.get("ipadapter_start_at_max", start_static)
                end_static = ipadapter_data["end_at"]
                end_min = wave_config.get("ipadapter_end_at_min", 1.0)
                end_max = wave_config.get("ipadapter_end_at_max", end_static)
                
                weight_A = w_max * (1.0 - t) + w_min * t
                start_A = start_max * (1.0 - t) + start_min * t
                end_A = end_max * (1.0 - t) + end_min * t
                weight_B = w_min * (1.0 - t) + w_max * t
                start_B = start_min * (1.0 - t) + start_max * t
                end_B = end_min * (1.0 - t) + end_max * t
                
                current_model = model.clone()
                if weight_A > 0:
                    current_model, _, _ = tiled_processor.apply_tiled(
                        current_model,
                        ipadapter_data["ipadapter"],
                        image=ipadapter_data["frames_A"][i],
                        weight=weight_A,
                        weight_type=ipadapter_data["weight_type"],
                        start_at=start_A,
                        end_at=end_A,
                        sharpening=0.0,
                        combine_embeds=ipadapter_data["combine_embeds"],
                        image_negative=ipadapter_data["image_negative"],
                        attn_mask=ipadapter_data["attn_mask"],
                        clip_vision=ipadapter_data["clipvision"],
                        embeds_scaling=ipadapter_data["embeds_scaling"],
                    )
                if weight_B > 0:
                    current_model, _, _ = tiled_processor.apply_tiled(
                        current_model,
                        ipadapter_data["ipadapter"],
                        image=ipadapter_data["frames_B"][i],
                        weight=weight_B,
                        weight_type=ipadapter_data["weight_type"],
                        start_at=start_B,
                        end_at=end_B,
                        sharpening=0.0,
                        combine_embeds=ipadapter_data["combine_embeds"],
                        image_negative=ipadapter_data["image_negative"],
                        attn_mask=ipadapter_data["attn_mask"],
                        clip_vision=ipadapter_data["clipvision"],
                        embeds_scaling=ipadapter_data["embeds_scaling"],
                    )

            # Denoise
            denoise_min = wave_config.get("denoise_min", 0.1)
            denoise_max = wave_config.get("denoise_max", 0.6)
            current_wave_type = wave_config.get("wave_type", "bell_curve")
            base_denoise = calculate_wave(current_wave_type, i, overlap_frames, denoise_min, denoise_max)
            ramp_len = max(1, int(overlap_frames * 0.1))
            if i < ramp_len:
                edge_factor = i / ramp_len
            elif i >= overlap_frames - ramp_len:
                edge_factor = (overlap_frames - 1 - i) / ramp_len
            else:
                edge_factor = 1.0
            denoise = base_denoise * edge_factor

            print(f"Morph Frame {i+1}/{overlap_frames}: Blend={t:.2f}, Denoise={denoise:.2f}, Steps={steps}")

            if denoise <= 0.0:
                processed_morph_frames_list.append(blended_latent)
                previous_latent = blended_latent
                continue
            
            latent_dict = {"samples": blended_latent}
            result = nodes.common_ksampler(
                current_model, seed, steps, cfg, sampler_name, scheduler,
                current_positive, current_negative, latent_dict,
                denoise=denoise
            )
            processed_morph_frames_list.append(result[0]["samples"])
            previous_latent = result[0]["samples"]
        
        morph_tensor = torch.cat(processed_morph_frames_list, dim=0)

        # Final stitch
        final_output = torch.cat([seg1_A, morph_tensor, seg3_B], dim=0)
        
        comfy.model_management.soft_empty_cache()
        
        return ({"samples": final_output},)


NODE_CLASS_MAPPINGS = {
    "VideoInterpolateSampler": VideoInterpolateSampler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoInterpolateSampler": "Video Interpolate Sampler"
}