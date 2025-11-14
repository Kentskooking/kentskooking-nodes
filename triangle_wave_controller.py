import torch
import math

class TriangleWaveController:
    """
    Generates triangle wave values for controlling video frame parameters.
    Creates smooth oscillations for denoise, steps, zoom, and CLIP vision strength.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cycle_length": ("INT", {"default": 100, "min": 1, "max": 10000, "step": 1}),

                "steps_min": ("INT", {"default": 16, "min": 1, "max": 10000, "step": 1}),
                "steps_max": ("INT", {"default": 20, "min": 1, "max": 10000, "step": 1}),

                "denoise_min": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "denoise_max": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),

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
        """
        Calculate triangle wave value at given position.

        Args:
            position: Current frame in cycle (0 to cycle_length)
            cycle_length: Length of one complete cycle
            min_val: Minimum value (at start/end of cycle)
            max_val: Maximum value (at peak of cycle)

        Returns:
            Float value between min_val and max_val
        """
        half_cycle = cycle_length / 2.0

        if position <= half_cycle:
            t = position / half_cycle
        else:
            t = (cycle_length - position) / half_cycle

        return min_val + (max_val - min_val) * t

    def create_config(self, cycle_length,
                      steps_min, steps_max,
                      denoise_min, denoise_max,
                      zoom_min, zoom_max,
                      clip_strength_min, clip_strength_max):
        """
        Create configuration dictionary for triangle wave parameters.
        The Video Iterative Sampler will use this to calculate values per frame.
        """
        config = {
            "cycle_length": cycle_length,
            "steps_min": steps_min,
            "steps_max": steps_max,
            "denoise_min": denoise_min,
            "denoise_max": denoise_max,
            "zoom_min": zoom_min,
            "zoom_max": zoom_max,
            "clip_strength_min": clip_strength_min,
            "clip_strength_max": clip_strength_max,
            "triangle_wave_func": self.triangle_wave,
        }
        return (config,)

    def calculate_for_frame(self, frame_idx, config):
        """
        Calculate all parameter values for a specific frame using triangle waves.
        Called by Video Iterative Sampler.
        """
        position_in_cycle = frame_idx % config["cycle_length"]

        steps_float = self.triangle_wave(position_in_cycle, config["cycle_length"],
                                         config["steps_min"], config["steps_max"])
        steps = int(round(steps_float))

        denoise = self.triangle_wave(position_in_cycle, config["cycle_length"],
                                     config["denoise_min"], config["denoise_max"])

        zoom_factor = self.triangle_wave(position_in_cycle, config["cycle_length"],
                                         config["zoom_min"], config["zoom_max"])

        clip_strength = self.triangle_wave(position_in_cycle, config["cycle_length"],
                                           config["clip_strength_min"], config["clip_strength_max"])

        return steps, denoise, zoom_factor, clip_strength


NODE_CLASS_MAPPINGS = {
    "TriangleWaveController": TriangleWaveController
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TriangleWaveController": "Triangle Wave Controller"
}
