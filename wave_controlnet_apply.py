class WaveControlNetApply:
    """
    ControlNet apply node that stores controlnet data in wave_config
    while forwarding the conditioning stream unchanged for chaining.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "control_net": ("CONTROL_NET",),
                "image": ("IMAGE",),
                "wave_config": ("TRIANGLE_WAVE_CONFIG",),
            },
            "optional": {
                "vae": ("VAE",),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "TRIANGLE_WAVE_CONFIG",)
    RETURN_NAMES = ("positive", "negative", "wave_config",)
    FUNCTION = "attach_controlnet"
    CATEGORY = "kentskooking/controllers"

    def attach_controlnet(self, positive, negative, control_net, image, wave_config, vae=None):
        """
        Clone the wave config, attach controlnet metadata, and return it alongside the
        passthrough conditioning so multiple wave-aware nodes can be chained.
        """
        config = dict(wave_config)

        # Convert image to control_hint format (move channels dimension)
        control_hint = image.movedim(-1, 1)

        config["controlnet_model"] = control_net
        config["controlnet_hint"] = control_hint
        config["controlnet_vae"] = vae

        return (positive, negative, config)


NODE_CLASS_MAPPINGS = {
    "WaveControlNetApply": WaveControlNetApply
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WaveControlNetApply": "Wave ControlNet Apply"
}
