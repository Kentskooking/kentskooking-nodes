WEIGHT_TYPES = ["linear", "ease in", "ease out", 'ease in-out', 'reverse in-out', 'weak input', 'weak output', 'weak middle', 'strong middle', 'style transfer', 'composition', 'strong style transfer']

class WaveIPAdapterAdvanced:
    """
    IPAdapter Advanced wrapper that stores configuration in wave_config.
    The actual IPAdapter application happens per-frame in the samplers.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "ipadapter": ("IPADAPTER",),
                "image": ("IMAGE",),
                "weight": ("FLOAT", {"default": 1.0, "min": -1, "max": 5, "step": 0.05}),
                "weight_type": (WEIGHT_TYPES,),
                "combine_embeds": (["concat", "add", "subtract", "average", "norm average"],),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "embeds_scaling": (['V only', 'K+V', 'K+V w/ C penalty', 'K+mean(V) w/ C penalty'],),
            },
            "optional": {
                "image_negative": ("IMAGE",),
                "attn_mask": ("MASK",),
                "clip_vision": ("CLIP_VISION",),
                "wave_config": ("WAVE_CONFIG",),
            }
        }

    RETURN_TYPES = ("MODEL", "WAVE_CONFIG")
    RETURN_NAMES = ("model", "wave_config")
    FUNCTION = "store_config"
    CATEGORY = "kentskooking/ipadapter"

    def store_config(self, model, ipadapter, image, weight, weight_type, combine_embeds,
                     start_at, end_at, embeds_scaling, wave_config=None,
                     image_negative=None, attn_mask=None, clip_vision=None):
        """
        Store IPAdapter configuration in wave_config for per-frame application.
        Model is passed through unchanged - application happens in the samplers.
        Supports image batches - will cycle through images at each cycle boundary.
        """
        if wave_config is None:
            raise Exception("WaveIPAdapterAdvanced requires a wave_config input. Connect a Video Wave Controller first.")

        enriched_config = wave_config.copy()

        # Split image batch into sequence for per-cycle selection
        image_sequence = self._split_image_batch(image)

        # Store all IPAdapter parameters needed for per-frame application
        enriched_config["ipadapter_model"] = ipadapter
        enriched_config["ipadapter_image"] = image  # Store full batch as fallback
        enriched_config["ipadapter_image_sequence"] = image_sequence  # Store split sequence
        enriched_config["ipadapter_image_sequence_length"] = len(image_sequence)
        enriched_config["ipadapter_weight"] = weight  # default weight, can be overridden by wave params
        enriched_config["ipadapter_weight_type"] = weight_type
        enriched_config["ipadapter_combine_embeds"] = combine_embeds
        enriched_config["ipadapter_start_at"] = start_at  # default, can be overridden
        enriched_config["ipadapter_end_at"] = end_at  # default, can be overridden
        enriched_config["ipadapter_embeds_scaling"] = embeds_scaling
        enriched_config["ipadapter_clip_vision"] = clip_vision
        enriched_config["ipadapter_image_negative"] = image_negative
        enriched_config["ipadapter_attn_mask"] = attn_mask

        # Mark that IPAdapter is enabled
        enriched_config["ipadapter_enabled"] = True

        return (model, enriched_config)

    def _split_image_batch(self, image):
        """
        Split an image batch into a list of individual images.
        Each image in the sequence will be used for one complete cycle.
        """
        import torch
        if image.shape[0] == 1:
            return [image]

        sequence = []
        for idx in range(image.shape[0]):
            sequence.append(image[idx:idx+1].clone())

        return sequence


NODE_CLASS_MAPPINGS = {
    "WaveIPAdapterAdvanced": WaveIPAdapterAdvanced
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WaveIPAdapterAdvanced": "Wave IPAdapter Advanced"
}