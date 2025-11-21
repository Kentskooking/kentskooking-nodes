class ImageWaveController:
    """
    Controller specifically designed for ImageIterativeSampler.
    Defines cycles, step ranges, and zoom rates for progressive image morphing.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cycle_count": ("INT", {"default": 5, "min": 1, "max": 10000, "step": 1, "tooltip": "Number of times to repeat the morph cycle"}),
                "step_floor": ("INT", {"default": 5, "min": 1, "max": 10000, "step": 1, "tooltip": "Minimum steps to run (start of cycle)"}),
                "start_at_step": ("INT", {"default": 16, "min": 0, "max": 10000, "step": 1, "tooltip": "Step scheduler start point"}),
                "end_at_step": ("INT", {"default": 36, "min": 1, "max": 10000, "step": 1, "tooltip": "Step scheduler end point"}),
                "zoom_rate": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.01, "tooltip": "Exponential zoom multiplier per iteration (e.g. 1.05 = 5% zoom)"}),
                "clip_strength_min": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "clip_strength_max": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("WAVE_CONFIG",)
    RETURN_NAMES = ("wave_config",)
    FUNCTION = "create_config"
    CATEGORY = "kentskooking/controllers"

    def create_config(self, cycle_count, step_floor, start_at_step, end_at_step,
                      zoom_rate, clip_strength_min, clip_strength_max):
        
        step_floor = max(1, step_floor)
        if end_at_step <= start_at_step:
            end_at_step = start_at_step + step_floor

        # Calculate iterations per cycle for compatibility with Visualizer
        iterations_per_cycle = (end_at_step - start_at_step) - step_floor + 1
        iterations_per_cycle = max(1, iterations_per_cycle)

        config = {
            "controller_type": "image",
            "cycle_count": cycle_count,
            "cycle_length": iterations_per_cycle, # Mapped for Visualizer compatibility
            "step_floor": step_floor,
            "start_at_step": start_at_step,
            "end_at_step": end_at_step,
            "zoom_rate": zoom_rate,
            "clip_strength_min": clip_strength_min,
            "clip_strength_max": clip_strength_max,
            # For visualizer to show simple ramp behavior if needed
            "wave_type": "sawtooth", 
        }
        return (config,)


NODE_CLASS_MAPPINGS = {
    "ImageWaveController": ImageWaveController
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageWaveController": "Image Wave Controller"
}
