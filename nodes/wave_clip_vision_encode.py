import torch
from comfy.clip_vision import Output as ClipVisionOutput


class WaveClipVisionEncode:
    """
    CLIPVisionEncode variant that also enriches the wave_config with a sequence
    of CLIP vision embeddings for cycle-based selection downstream.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_vision": ("CLIP_VISION",),
                "image": ("IMAGE",),
                "crop": (["center", "none"], {"default": "center"}),
                "wave_config": ("WAVE_CONFIG",),
            }
        }

    RETURN_TYPES = ("CLIP_VISION_OUTPUT", "WAVE_CONFIG")
    RETURN_NAMES = ("clip_vision_output", "wave_config")
    FUNCTION = "encode_with_wave_config"
    CATEGORY = "kentskooking/controllers"

    def encode_with_wave_config(self, clip_vision, image, crop, wave_config):
        """
        Encode one or more images and attach the resulting embedding sequence to
        the wave configuration for downstream per-cycle selection.
        """
        crop_image = crop == "center"
        batch_size = image.shape[0]
        chunk_size = 16

        # Process in chunks to reduce VRAM usage
        if batch_size <= chunk_size:
            clip_output = clip_vision.encode_image(image, crop=crop_image)
        else:
            chunk_outputs = []
            for i in range(0, batch_size, chunk_size):
                chunk = image[i:i+chunk_size]
                chunk_output = clip_vision.encode_image(chunk, crop=crop_image)
                chunk_outputs.append(chunk_output)

            # Merge chunk outputs into single ClipVisionOutput
            clip_output = self._merge_clip_outputs(chunk_outputs)

        config = dict(wave_config)

        sequence = self._split_clip_output(clip_output)
        config["clip_vision_sequence"] = sequence
        config["clip_vision_sequence_wrap"] = True
        config["clip_vision_sequence_length"] = len(sequence)
        config["clip_vision_output"] = clip_output

        return (clip_output, config)

    def _merge_clip_outputs(self, outputs):
        """
        Merge multiple ClipVisionOutput objects into a single batched output.
        """
        if len(outputs) == 1:
            return outputs[0]

        merged = ClipVisionOutput()
        attrs = list(vars(outputs[0]).keys())

        for key in attrs:
            values = [getattr(out, key) for out in outputs]
            if isinstance(values[0], torch.Tensor):
                # Concatenate tensors along batch dimension
                merged[key] = torch.cat(values, dim=0)
            else:
                # Non-tensor attributes: use first value
                merged[key] = values[0]

        return merged

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
            return []

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
    "WaveClipVisionEncode": WaveClipVisionEncode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WaveClipVisionEncode": "Wave CLIP Vision Encode"
}
