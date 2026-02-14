import torch
import comfy.sample
import comfy.samplers
import comfy.utils
import comfy.model_management
import latent_preview
import inspect
from ..utils.kentskooking_utils import (
    calculate_explorer_path_strengths,
    build_explorer_conditioning_multi,
    repeat_conditioning_to_batch,
    stack_conditioning_batches,
)


class ExplorerConditioningSampler:
    """
    Explores latent output across weighted concat traversals between
    positive conditioning branches while keeping the seed fixed.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive_a": ("CONDITIONING",),
                "positive_b": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "wave_config": ("WAVE_CONFIG",),
                "iteration_count": ("INT", {"default": 8, "min": 1, "max": 10000, "step": 1}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 1024, "step": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "step": 1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "positive_c": ("CONDITIONING",),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent_batch",)
    FUNCTION = "process_exploration"
    CATEGORY = "kentskooking/sampling"

    def _get_curve_type(self, wave_config):
        curve_type = (wave_config or {}).get("curve_type", "linear")
        if curve_type not in {"linear", "sine", "ease_in_out"}:
            return "linear"
        return curve_type

    def _get_strength_range(self, wave_config):
        cfg = wave_config or {}
        return (
            float(cfg.get("positive_a_start", 0.0)),
            float(cfg.get("positive_a_end", 1.0)),
            float(cfg.get("positive_b_start", 1.0)),
            float(cfg.get("positive_b_end", 0.0)),
            float(cfg.get("positive_c_start", 0.0)),
            float(cfg.get("positive_c_end", 0.0)),
        )

    def _sampler_supports_noise_sampler(self, sampler_name):
        sampler_impl = comfy.samplers.ksampler(sampler_name)
        signature = inspect.signature(sampler_impl.sampler_function)
        return "noise_sampler" in signature.parameters

    def _build_identical_noise_sampler(self, noise_ref, seed, target_device):
        """
        Build a noise sampler that returns identical stochastic noise across
        all batch members for each stochastic sampling step.
        """
        device = target_device
        if not isinstance(device, torch.device):
            device = torch.device(device)

        local_seed = int(seed)
        if device.type == "cpu":
            local_seed += 1

        generator = torch.Generator(device=device)
        generator.manual_seed(local_seed)

        sample_shape = [1] + list(noise_ref.shape[1:])
        repeats = [noise_ref.shape[0]] + [1] * (noise_ref.ndim - 1)
        dtype = noise_ref.dtype
        layout = noise_ref.layout

        def noise_sampler(sigma, sigma_next):
            base = torch.randn(sample_shape, dtype=dtype, layout=layout, device=device, generator=generator)
            sampled = base.repeat(*repeats)
            if torch.is_tensor(sigma) and sampled.device != sigma.device:
                return sampled.to(sigma.device)
            return sampled

        return noise_sampler

    def _run_common_ksampler_with_identical_batch_noise(self, model, seed, steps, cfg, sampler_name, scheduler,
                                                        positive, negative, latent, denoise):
        """
        Mirrors nodes.common_ksampler while allowing a custom per-step
        stochastic noise sampler so all batch members stay noise-locked.
        """
        latent_image = latent["samples"]
        latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image, latent.get("downscale_ratio_spacial", None))

        batch_inds = latent["batch_index"] if "batch_index" in latent else None
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

        noise_mask = latent.get("noise_mask", None)

        callback = latent_preview.prepare_callback(model, steps)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        # Build sampler with identical stochastic step-noise when supported.
        if self._sampler_supports_noise_sampler(sampler_name):
            identical_noise_sampler = self._build_identical_noise_sampler(noise, seed, model.load_device)
            sampler_obj = comfy.samplers.ksampler(sampler_name, extra_options={"noise_sampler": identical_noise_sampler})
        else:
            sampler_obj = comfy.samplers.sampler_object(sampler_name)

        sigma_calc = comfy.samplers.KSampler(
            model,
            steps=steps,
            device=model.load_device,
            sampler=sampler_name,
            scheduler=scheduler,
            denoise=denoise,
            model_options=model.model_options,
        )
        sigmas = sigma_calc.sigmas

        samples = comfy.samplers.sample(
            model,
            noise,
            positive,
            negative,
            cfg,
            model.load_device,
            sampler_obj,
            sigmas,
            model.model_options,
            latent_image=latent_image,
            denoise_mask=noise_mask,
            callback=callback,
            disable_pbar=disable_pbar,
            seed=seed,
        )
        samples = samples.to(comfy.model_management.intermediate_device())

        out = latent.copy()
        out.pop("downscale_ratio_spacial", None)
        out["samples"] = samples
        return (out,)

    def process_exploration(self, model, positive_a, positive_b, negative, latent_image, wave_config,
                            iteration_count, batch_size, seed, steps, sampler_name, scheduler, cfg, denoise,
                            positive_c=None):
        base_latent = latent_image["samples"]
        curve_type = self._get_curve_type(wave_config)
        positive_a_start, positive_a_end, positive_b_start, positive_b_end, positive_c_start, positive_c_end = self._get_strength_range(wave_config)
        c_range_active = abs(positive_c_start) > 1e-6 or abs(positive_c_end) > 1e-6
        has_positive_c = (positive_c is not None) and c_range_active
        loop_requested = bool((wave_config or {}).get("loop_video", False))
        use_loop = loop_requested and has_positive_c

        if base_latent.shape[0] != 1:
            raise Exception(
                f"ExplorerConditioningSampler expects a single latent input (batch size = 1), got {base_latent.shape[0]}"
            )

        print(f"\n{'='*60}")
        print(f"ExplorerConditioningSampler: {iteration_count} iterations")
        print(f"  Batch size: {batch_size}")
        print(f"  Curve: {curve_type}")
        print(f"  Positive A range: {positive_a_start} -> {positive_a_end}")
        print(f"  Positive B range: {positive_b_start} -> {positive_b_end}")
        if has_positive_c:
            print(f"  Positive C range: {positive_c_start} -> {positive_c_end}")
            print(f"  Path: A->B->C{'->A' if use_loop else ''}")
        elif positive_c is not None and not c_range_active:
            print("  Positive C is connected but C range is 0 -> 0; running A->B only")
        elif loop_requested:
            print("  Loop mode requested but positive_c is not connected; running A->B only")
        print(f"  Seed (locked): {seed}")
        print(f"{'='*60}\n")

        processed_latents = []
        progress_interval = max(1, iteration_count // 10)
        negative_cache = {}

        for chunk_start in range(0, iteration_count, batch_size):
            chunk_end = min(iteration_count, chunk_start + batch_size)
            chunk_len = chunk_end - chunk_start

            conditionings_for_chunk = []
            strengths_for_chunk = []
            for iteration_idx in range(chunk_start, chunk_end):
                strength_a, strength_b, strength_c = calculate_explorer_path_strengths(
                    iteration_idx=iteration_idx,
                    iteration_count=iteration_count,
                    curve_type=curve_type,
                    positive_a_start=positive_a_start,
                    positive_a_end=positive_a_end,
                    positive_b_start=positive_b_start,
                    positive_b_end=positive_b_end,
                    positive_c_start=positive_c_start,
                    positive_c_end=positive_c_end,
                    has_positive_c=has_positive_c,
                    loop_video=use_loop,
                )

                branch_conditionings = [positive_a, positive_b]
                branch_strengths = [strength_a, strength_b]
                if has_positive_c:
                    branch_conditionings.append(positive_c)
                    branch_strengths.append(strength_c)

                strengths_for_chunk.append((strength_a, strength_b, strength_c))
                conditionings_for_chunk.append(
                    build_explorer_conditioning_multi(branch_conditionings, branch_strengths)
                )

            iteration_positive = stack_conditioning_batches(conditionings_for_chunk)
            if chunk_len not in negative_cache:
                negative_cache[chunk_len] = repeat_conditioning_to_batch(negative, chunk_len)
            iteration_negative = negative_cache[chunk_len]

            latent_dict = {
                "samples": base_latent.repeat(chunk_len, 1, 1, 1),
                # Force same initial noise for all samples in this chunk.
                "batch_index": [0] * chunk_len,
            }
            result = self._run_common_ksampler_with_identical_batch_noise(
                model=model,
                seed=seed,
                steps=steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                positive=iteration_positive,
                negative=iteration_negative,
                latent=latent_dict,
                denoise=denoise,
            )

            processed_latents.append(result[0]["samples"])

            if iteration_count <= 20:
                for local_idx, (strength_a, strength_b, strength_c) in enumerate(strengths_for_chunk):
                    iter_number = chunk_start + local_idx + 1
                    if has_positive_c:
                        print(
                            f"Iteration {iter_number}/{iteration_count}: "
                            f"strength_a={strength_a:.3f}, strength_b={strength_b:.3f}, strength_c={strength_c:.3f}"
                        )
                    else:
                        print(
                            f"Iteration {iter_number}/{iteration_count}: "
                            f"strength_a={strength_a:.3f}, strength_b={strength_b:.3f}"
                        )
            else:
                first_a, first_b, first_c = strengths_for_chunk[0]
                last_a, last_b, last_c = strengths_for_chunk[-1]
                if chunk_start == 0 or chunk_end == iteration_count or chunk_end % progress_interval == 0:
                    if has_positive_c:
                        print(
                            f"Chunk {chunk_start + 1}-{chunk_end}/{iteration_count}: "
                            f"A {first_a:.3f}->{last_a:.3f}, "
                            f"B {first_b:.3f}->{last_b:.3f}, "
                            f"C {first_c:.3f}->{last_c:.3f}"
                        )
                    else:
                        print(
                            f"Chunk {chunk_start + 1}-{chunk_end}/{iteration_count}: "
                            f"A {first_a:.3f}->{last_a:.3f}, B {first_b:.3f}->{last_b:.3f}"
                        )

        all_latents = torch.cat(processed_latents, dim=0)
        comfy.model_management.soft_empty_cache()
        return ({"samples": all_latents},)


NODE_CLASS_MAPPINGS = {
    "ExplorerConditioningSampler": ExplorerConditioningSampler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ExplorerConditioningSampler": "Explorer Conditioning Sampler"
}
