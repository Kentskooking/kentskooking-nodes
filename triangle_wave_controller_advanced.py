class TriangleWaveControllerAdvanced:
    """
    Triangle wave controller focusing on advanced sampling ranges.
    Provides cycle-based modulation for step count, zoom, and CLIP strength
    while tracking start/end step metadata for future KSamplerAdvanced usage.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cycle_length": ("INT", {"default": 60, "min": 1, "max": 10000, "step": 1}),
                "step_floor": ("INT", {"default": 5, "min": 1, "max": 10000, "step": 1}),
                "start_at_step": ("INT", {"default": 16, "min": 0, "max": 10000, "step": 1}),
                "end_at_step": ("INT", {"default": 36, "min": 1, "max": 10000, "step": 1}),
                "zoom_min": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.01}),
                "zoom_max": ("FLOAT", {"default": 1.25, "min": 0.1, "max": 10.0, "step": 0.01}),
                "clip_strength_min": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "clip_strength_max": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("TRIANGLE_WAVE_CONFIG",)
    RETURN_NAMES = ("wave_config",)
    FUNCTION = "create_config"
    CATEGORY = "kentskooking/controllers"

    def triangle_wave(self, position, cycle_length, min_val, max_val):
        """Calculate triangle wave value at given position."""
        half_cycle = cycle_length / 2.0
        if position <= half_cycle:
            t = position / half_cycle
        else:
            t = (cycle_length - position) / half_cycle
        return min_val + (max_val - min_val) * t

    def create_config(self, cycle_length, step_floor, start_at_step, end_at_step,
                      zoom_min, zoom_max, clip_strength_min, clip_strength_max):
        """
        Create configuration dictionary for triangle wave parameters.
        The triangle wave will oscillate the END step between start+floor and end_at_step.
        """
        step_floor = max(1, step_floor)
        if end_at_step <= start_at_step:
            end_at_step = start_at_step + step_floor

        config = {
            "cycle_length": cycle_length,
            "step_floor": step_floor,
            "start_at_step": start_at_step,
            "end_at_step": end_at_step,
            "zoom_min": zoom_min,
            "zoom_max": zoom_max,
            "clip_strength_min": clip_strength_min,
            "clip_strength_max": clip_strength_max,
            "controller_variant": "advanced",
        }
        return (config,)

    def calculate_for_frame(self, frame_idx, config):
        """
        Calculate per-frame step count and end step using a triangle wave.
        The start step stays fixed; the wave modulates the end step.
        """
        position = frame_idx % config["cycle_length"]
        step_floor = max(1, config["step_floor"])
        start_at = config["start_at_step"]
        max_end = max(start_at + step_floor, config["end_at_step"])
        min_end = start_at + step_floor

        current_end = self.triangle_wave(position, config["cycle_length"], min_end, max_end)
        end_at = int(round(max(current_end, min_end)))
        steps = max(step_floor, end_at - start_at)

        zoom = self.triangle_wave(position, config["cycle_length"], config["zoom_min"], config["zoom_max"])
        clip_strength = self.triangle_wave(position, config["cycle_length"],
                                           config["clip_strength_min"], config["clip_strength_max"])

        return steps, start_at, end_at, zoom, clip_strength


NODE_CLASS_MAPPINGS = {
    "TriangleWaveControllerAdvanced": TriangleWaveControllerAdvanced
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TriangleWaveControllerAdvanced": "Triangle Wave Controller (Advanced)"
}
