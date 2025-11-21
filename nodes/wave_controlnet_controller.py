class WaveControlNetController:
    """
    Enriches triangle wave config with ControlNet-specific parameters.
    Modulates ControlNet strength, start_at, and end_at based on existing cycle_length.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "wave_config": ("WAVE_CONFIG",),
                "strength_min": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.01}),
                "strength_max": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "start_at_min": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "start_at_max": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_at_min": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_at_max": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("WAVE_CONFIG",)
    RETURN_NAMES = ("wave_config",)
    FUNCTION = "enrich_config"
    CATEGORY = "kentskooking/controllers"

    def triangle_wave(self, position, cycle_length, min_val, max_val):
        """Calculate triangle wave value at given position."""
        half_cycle = cycle_length / 2.0
        if position <= half_cycle:
            t = position / half_cycle
        else:
            t = (cycle_length - position) / half_cycle
        return min_val + (max_val - min_val) * t

    def enrich_config(self, wave_config, strength_min, strength_max,
                      start_at_min, start_at_max, end_at_min, end_at_max):
        """
        Add ControlNet parameters to the wave config.
        The samplers will use these to modulate ControlNet per frame.
        """
        enriched_config = wave_config.copy()

        enriched_config["controlnet_strength_min"] = strength_min
        enriched_config["controlnet_strength_max"] = strength_max
        enriched_config["controlnet_start_at_min"] = start_at_min
        enriched_config["controlnet_start_at_max"] = start_at_max
        enriched_config["controlnet_end_at_min"] = end_at_min
        enriched_config["controlnet_end_at_max"] = end_at_max

        return (enriched_config,)

    def calculate_for_frame(self, frame_idx, config):
        """
        Calculate ControlNet parameters for a specific frame.
        Called by Video Iterative Samplers.
        """
        position = frame_idx % config["cycle_length"]

        strength = self.triangle_wave(position, config["cycle_length"],
                                     config["controlnet_strength_min"],
                                     config["controlnet_strength_max"])

        start_at = self.triangle_wave(position, config["cycle_length"],
                                     config["controlnet_start_at_min"],
                                     config["controlnet_start_at_max"])

        end_at = self.triangle_wave(position, config["cycle_length"],
                                   config["controlnet_end_at_min"],
                                   config["controlnet_end_at_max"])

        return strength, start_at, end_at


NODE_CLASS_MAPPINGS = {
    "WaveControlNetController": WaveControlNetController
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WaveControlNetController": "Wave ControlNet Controller"
}
