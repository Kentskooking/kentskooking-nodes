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
                "wave_config": ("WAVE_CONFIG",),
            },
            "optional": {
                "vae": ("VAE",),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "WAVE_CONFIG",)
    RETURN_NAMES = ("positive", "negative", "wave_config",)
    FUNCTION = "attach_controlnet"
    CATEGORY = "kentskooking/controllers"

    def attach_controlnet(self, positive, negative, control_net, image, wave_config, vae=None):
        """
        Clone the wave config, attach controlnet metadata, and return it alongside the
        passthrough conditioning so multiple wave-aware nodes can be chained.
        
        Also attaches the controlnet data directly to the conditioning dictionary
        so the VideoInterpolateSampler can retrieve it per-stream.
        """
        config = dict(wave_config)

        # Convert image to control_hint format (move channels dimension)
        control_hint = image.movedim(-1, 1)

        config["controlnet_model"] = control_net
        config["controlnet_hint"] = control_hint
        config["controlnet_vae"] = vae
        
        # Create a self-contained controlnet packet to attach to the conditioning
        # This allows VideoInterpolateSampler to see specific CNs for Stream A vs Stream B
        cnet_packet = {
            "control_net": control_net,
            "control_hint": control_hint,
            "vae": vae,
            "wave_config": config # Store the current wave config (with strength/start/end params)
        }

        # Helper to attach to conditioning list [[embedding, {dict}], ...]
        def attach_to_cond(cond_list):
            new_cond = []
            for t in cond_list:
                d = t[1].copy()
                d["kentskooking_controlnet"] = cnet_packet
                new_cond.append([t[0], d])
            return new_cond

        new_positive = attach_to_cond(positive)
        new_negative = attach_to_cond(negative)

        return (new_positive, new_negative, config)


NODE_CLASS_MAPPINGS = {
    "WaveControlNetApply": WaveControlNetApply
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WaveControlNetApply": "Wave ControlNet Apply"
}
