from ..utils.kentskooking_utils import calculate_wave

class VideoWaveController:
    """
    Triangle wave controller focusing on advanced sampling ranges.
    Provides cycle-based modulation for step count, zoom, and CLIP strength
    while tracking start/end step metadata for future KSamplerAdvanced usage.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "wave_type": (["triangle", "sine", "sawtooth", "square"], {"default": "triangle"}),
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

    RETURN_TYPES = ("WAVE_CONFIG",)
    RETURN_NAMES = ("wave_config",)
    FUNCTION = "create_config"
    CATEGORY = "kentskooking/controllers"

    def create_config(self, wave_type, cycle_length, step_floor, start_at_step, end_at_step,
                      zoom_min, zoom_max, clip_strength_min, clip_strength_max):
        """
        Create configuration dictionary for triangle wave parameters.
        The triangle wave will oscillate the END step between start+floor and end_at_step.
        """
        step_floor = max(1, step_floor)
        if end_at_step <= start_at_step:
            end_at_step = start_at_step + step_floor

        config = {
            "wave_type": wave_type,
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
        wave_type = config.get("wave_type", "triangle")
        position = frame_idx % config["cycle_length"]
        step_floor = max(1, config["step_floor"])
        start_at = config["start_at_step"]
        max_end = max(start_at + step_floor, config["end_at_step"])
        min_end = start_at + step_floor

        current_end = calculate_wave(wave_type, position, config["cycle_length"], min_end, max_end)
        end_at = int(round(max(current_end, min_end)))
        steps = max(step_floor, end_at - start_at)

        zoom = calculate_wave(wave_type, position, config["cycle_length"], config["zoom_min"], config["zoom_max"])
        clip_strength = calculate_wave(wave_type, position, config["cycle_length"],
                                           config["clip_strength_min"], config["clip_strength_max"])

        return steps, start_at, end_at, zoom, clip_strength


NODE_CLASS_MAPPINGS = {
    "VideoWaveController": VideoWaveController
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoWaveController": "Video Wave Controller"
}
