class WaveStyleModelApply:
    """
    StyleModelApply-compatible node that enriches the wave config with style data
    while forwarding the conditioning stream unchanged for chaining.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "style_model": ("STYLE_MODEL",),
                "strength_type": (["multiply", "attn_bias"], {"default": "multiply"}),
                "clip_vision_output": ("CLIP_VISION_OUTPUT",),
                "wave_config": ("TRIANGLE_WAVE_CONFIG",),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "TRIANGLE_WAVE_CONFIG",)
    RETURN_NAMES = ("conditioning", "wave_config",)
    FUNCTION = "attach_style_model"
    CATEGORY = "kentskooking/controllers"

    def attach_style_model(self, conditioning, style_model, strength_type, clip_vision_output, wave_config):
        """
        Clone the wave config, attach style metadata, and return it alongside the
        passthrough conditioning so multiple wave-aware nodes can be chained.
        """
        config = dict(wave_config)
        config["style_model"] = style_model
        if clip_vision_output is not None:
            config["clip_vision_output"] = clip_vision_output
            if "clip_vision_sequence" not in config:
                config["clip_vision_sequence"] = [clip_vision_output]
                config["clip_vision_sequence_length"] = 1
                config["clip_vision_sequence_wrap"] = True
        config["style_strength_type"] = strength_type
        return (conditioning, config)


NODE_CLASS_MAPPINGS = {
    "WaveStyleModelApply": WaveStyleModelApply
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WaveStyleModelApply": "Wave Style Model Apply"
}
