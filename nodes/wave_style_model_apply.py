import torch
from comfy.clip_vision import Output as ClipVisionOutput

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
                "wave_config": ("WAVE_CONFIG",),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "WAVE_CONFIG",)
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
                # Split batch into sequence if not already present
                sequence = self._split_clip_output(clip_vision_output)
                config["clip_vision_sequence"] = sequence
                config["clip_vision_sequence_length"] = len(sequence)
                config["clip_vision_sequence_wrap"] = True
        config["style_strength_type"] = strength_type
        return (conditioning, config)

    def _split_clip_output(self, clip_output):
        """
        Break a batched CLIP_VISION_OUTPUT into a list of single-sample outputs.
        Each entry preserves the attribute-based API StyleModel expects.
        """
        attrs = list(vars(clip_output).keys())
        batch = None
        for key in attrs:
            value = getattr(clip_output, key)
            if isinstance(value, torch.Tensor) and value.dim() > 0:
                batch = value.shape[0]
                break

        if not batch:
            return [clip_output] # Fallback if no batch dim found

        sequence = []
        for idx in range(batch):
            single = ClipVisionOutput()
            for key in attrs:
                value = getattr(clip_output, key)
                if isinstance(value, torch.Tensor) and value.shape[0] == batch:
                    single[key] = value[idx:idx+1].clone()
                else:
                    single[key] = value
            sequence.append(single)

        return sequence


NODE_CLASS_MAPPINGS = {
    "WaveStyleModelApply": WaveStyleModelApply
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WaveStyleModelApply": "Wave Style Model Apply"
}
