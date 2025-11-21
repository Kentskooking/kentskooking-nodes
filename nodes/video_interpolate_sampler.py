import torch
import nodes
import comfy.samplers
import comfy.utils
import comfy.model_management
from ..utils.kentskooking_utils import calculate_wave, apply_controlnet_wrapper

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
        # Dim 0 is batch (frames), which can differ. Dims 1, 2, 3 must match.
        if tensor_A.shape[1:] != tensor_B.shape[1:]:
            raise ValueError(f"Latent Shape Mismatch! Video A: {tensor_A.shape}, Video B: {tensor_B.shape}. "
                             "Videos must have the same height, width, and channel count to be interpolated.")

        len_A = tensor_A.shape[0]
        len_B = tensor_B.shape[0]
        
        overlap_frames = wave_config.get("overlap_frames", 25)
        
        # Validation
        if len_A < overlap_frames or len_B < overlap_frames:
            raise ValueError(f"Video inputs too short! Overlap is {overlap_frames}, but videos are {len_A} and {len_B} frames.")

        # 1. Slice Segments
        # Segment 1: Video A (Start -> len_A - overlap) - untouched
        # Segment 2: Morph Zone (overlap frames)
        # Segment 3: Video B (overlap -> End) - untouched

        seg1_A = tensor_A[:-overlap_frames]

        morph_A = tensor_A[-overlap_frames:]
        morph_B = tensor_B[:overlap_frames]

        seg3_B = tensor_B[overlap_frames:]

        # Check for edge case: overlap equals video length (no untouched frames)
        if seg1_A.shape[0] == 0 and seg3_B.shape[0] == 0:
            print(f"\n⚠️  WARNING: Overlap ({overlap_frames}) equals video length - entire output will be morphed!")
            print(f"   Consider reducing overlap to preserve some untouched frames from each video.\n")

        print(f"\n{'='*60}")
        print(f"VideoInterpolateSampler: Stitching Seamless Video")
        print(f"  Part 1 (Source A): {seg1_A.shape[0]} frames")
        print(f"  Part 2 (Morph):    {overlap_frames} frames")
        print(f"  Part 3 (Source B): {seg3_B.shape[0]} frames")
        print(f"  Total Output:      {seg1_A.shape[0] + overlap_frames + seg3_B.shape[0]} frames")
        print(f"{'='*60}\n")

        processed_morph_frames = []

        # Initialize previous latent for feedback from the last frame of Segment 1 (A)
        # Fallback to first morph frame if seg1_A is empty (overlap == video length)
        if seg1_A.shape[0] > 0:
            previous_latent = seg1_A[-1:]
        else:
            # No untouched frames before morph zone - use first morph frame as initial reference
            previous_latent = morph_A[0:1]
        
        # 2. Process Morph Zone
        for i in range(overlap_frames):
            # Interpolation factor t (0.0 -> 1.0)
            t = i / (overlap_frames - 1) if overlap_frames > 1 else 0.5
            
            # 1. Blend Latents
            latent_a_frame = morph_A[i:i+1]
            latent_b_frame = morph_B[i:i+1]
            blended_latent = latent_a_frame * (1.0 - t) + latent_b_frame * t
            
            # Apply feedback: blend previous result into current noisy input
            # This stabilizes temporal flickering by dragging the input towards history
            if feedback_mode == "previous_frame":
                alpha = 0.1 # Hardcoded small feedback factor for stability
                blended_latent = blended_latent * (1.0 - alpha) + previous_latent * alpha
            
            # 2. Blend Conditioning (Base Embeddings)
            current_positive = self.interpolate_conditioning(positive_A, positive_B, t)
            current_negative = negative # Initialize negative conditioning for this frame
            
            # 3. Apply ControlNets dynamically
            # ControlNet A (Fading Out: Start at Max, End at Min)
            if len(positive_A) > 0 and "kentskooking_controlnet" in positive_A[0][1]:
                 cn_data = positive_A[0][1]["kentskooking_controlnet"]
                 # Frame Index: End of Video A + current morph index
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

                 # A fades OUT: t=0 is Max, t=1 is Min
                 strength = s_max * (1.0 - t) + s_min * t
                 start_at = start_max * (1.0 - t) + start_min * t
                 end_at = end_max * (1.0 - t) + end_min * t
                 
                 current_positive, current_negative = apply_controlnet_wrapper(
                     current_positive, current_negative,
                     cn_data["control_net"], hint, strength, start_at, end_at, vae=cn_data["vae"]
                 )

            # ControlNet B (Fading In: Start at Min, End at Max)
            if len(positive_B) > 0 and "kentskooking_controlnet" in positive_B[0][1]:
                 cn_data = positive_B[0][1]["kentskooking_controlnet"]
                 # Frame Index: Start of Video B (which is just i)
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

                 # B fades IN: t=0 is Min, t=1 is Max
                 strength = s_min * (1.0 - t) + s_max * t
                 start_at = start_min * (1.0 - t) + start_max * t
                 end_at = end_min * (1.0 - t) + end_max * t
                 
                 current_positive, current_negative = apply_controlnet_wrapper(
                     current_positive, current_negative,
                     cn_data["control_net"], hint, strength, start_at, end_at, vae=cn_data["vae"]
                 )

            # 4. Calculate Denoise (Bell Curve or Selected Wave)
            # Position 0..overlap. Bell curve peaks at overlap/2.
            denoise_min = wave_config.get("denoise_min", 0.1)
            denoise_max = wave_config.get("denoise_max", 0.6)
            current_wave_type = wave_config.get("wave_type", "bell_curve")
            
            # Use helper (defaults to bell_curve if wave_type is bell_curve in config)
            base_denoise = calculate_wave(current_wave_type, i, overlap_frames, denoise_min, denoise_max)
            
            # Seam Smoothing: Force denoise to 0 at the edges to prevent style jumps
            # We apply a window that is 0 at the ends and ramps to 1 quickly.
            # Ramp length = 10% of overlap (min 1 frame)
            ramp_len = max(1, int(overlap_frames * 0.1))
            
            if i < ramp_len:
                edge_factor = i / ramp_len
            elif i >= overlap_frames - ramp_len:
                edge_factor = (overlap_frames - 1 - i) / ramp_len
            else:
                edge_factor = 1.0
            
            denoise = base_denoise * edge_factor

            print(f"Morph Frame {i+1}/{overlap_frames}: Blend={t:.2f}, Denoise={denoise:.2f}, Steps={steps}")

            # Skip frames with zero denoise - no point running sampler
            if denoise <= 0.0:
                processed_morph_frames.append(blended_latent)
                previous_latent = blended_latent
                continue
            
            latent_dict = {"samples": blended_latent}
            
            # Run Sampler
            # Note: we use 'steps' as the total scheduler steps, but 'denoise' limits how much work is done.
            result = nodes.common_ksampler(
                model, seed, steps, cfg, sampler_name, scheduler,
                current_positive, current_negative, latent_dict,
                denoise=denoise
            )
            
            processed_morph_frames.append(result[0]["samples"])
            
            # Update previous_latent with the result of this frame for the next iteration's feedback
            previous_latent = result[0]["samples"]

        # 3. Concatenate
        morph_tensor = torch.cat(processed_morph_frames, dim=0)
        
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