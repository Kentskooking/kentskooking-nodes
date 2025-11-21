class InterpolationWaveController:
    """
    Controller for VideoInterpolateSampler.
    Defines the 'Morph Zone' parameters: overlap frames and denoising intensity.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "wave_type": (["bell_curve", "triangle", "sine", "sawtooth", "square"], {"default": "bell_curve"}),
                "overlap_frames": ("INT", {"default": 25, "min": 2, "max": 10000, "step": 1, "tooltip": "Number of frames to overlap and morph between videos"}),
                "denoise_min": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Denoise strength at start/end of morph"}),
                "denoise_max": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Peak denoise strength at middle of morph"}),
            }
        }

    RETURN_TYPES = ("WAVE_CONFIG",)
    RETURN_NAMES = ("wave_config",)
    FUNCTION = "create_config"
    CATEGORY = "kentskooking/controllers"

    def create_config(self, wave_type, overlap_frames, denoise_min, denoise_max):
        
        config = {
            "controller_type": "interpolation",
            "overlap_frames": overlap_frames,
            # Mapped for visualizer/utility compatibility (cycle_length = overlap zone)
            "cycle_length": overlap_frames, 
            "denoise_min": denoise_min,
            "denoise_max": denoise_max,
            "wave_type": wave_type,
        }
        return (config,)


NODE_CLASS_MAPPINGS = {
    "InterpolationWaveController": InterpolationWaveController
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InterpolationWaveController": "Interpolation Wave Controller"
}
