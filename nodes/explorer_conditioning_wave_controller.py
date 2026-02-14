class ExplorerConditioningWaveController:
    """
    Provides crossfade curve configuration for ExplorerConditioningSampler.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "curve_type": (["linear", "sine", "ease_in_out"], {"default": "linear"}),
                "positive_a_start": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "positive_a_end": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "positive_b_start": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "positive_b_end": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "positive_c_start": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "positive_c_end": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "loop_video": ("BOOLEAN", {"default": False, "label_on": "true", "label_off": "false"}),
            }
        }

    RETURN_TYPES = ("WAVE_CONFIG",)
    RETURN_NAMES = ("wave_config",)
    FUNCTION = "create_config"
    CATEGORY = "kentskooking/controllers"

    def create_config(self, curve_type, positive_a_start, positive_a_end, positive_b_start, positive_b_end, positive_c_start, positive_c_end, loop_video):
        config = {
            "controller_type": "explorer_conditioning",
            "curve_type": curve_type,
            "positive_a_start": positive_a_start,
            "positive_a_end": positive_a_end,
            "positive_b_start": positive_b_start,
            "positive_b_end": positive_b_end,
            "positive_c_start": positive_c_start,
            "positive_c_end": positive_c_end,
            "loop_video": loop_video,
        }
        return (config,)


NODE_CLASS_MAPPINGS = {
    "ExplorerConditioningWaveController": ExplorerConditioningWaveController
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ExplorerConditioningWaveController": "Explorer Conditioning Wave Controller"
}
