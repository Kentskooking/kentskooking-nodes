import torch
import os
from ..utils.kentskooking_utils import resolve_checkpoint_path, load_checkpoint_tensors
import folder_paths


class CheckpointPreviewLoader:
    """
    Load a .latent checkpoint file saved by ImageIterativeSampler or VideoIterativeSampler
    and expose its latent tensor as a LATENT output for video decoding.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "checkpoint_path": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": "",
                        "placeholder": "Path to .latent checkpoint file"
                    }
                ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "load_checkpoint"
    CATEGORY = "kentskooking/checkpointing"

    def load_checkpoint(self, checkpoint_path):
        if not checkpoint_path or not isinstance(checkpoint_path, str):
            raise Exception("CheckpointPreviewLoader: checkpoint_path must be a non-empty string.")

        # Default to ./output/video_checkpoints/ if no directory specified
        output_dir = folder_paths.get_output_directory()
        default_dir = os.path.join(output_dir, "video_checkpoints")

        # Resolve full path using shared utility
        try:
            final_path, _ = resolve_checkpoint_path(checkpoint_path, default_dir)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"CheckpointPreviewLoader: {e}")

        # Load tensors safely
        tensors, _ = load_checkpoint_tensors(final_path)

        if "latent_tensor" not in tensors:
            raise Exception(f"CheckpointPreviewLoader: 'latent_tensor' key not found in checkpoint. Available keys: {list(tensors.keys())}")

        samples = tensors["latent_tensor"]

        # Wrap in LATENT dict for ComfyUI
        return ({"samples": samples},)


NODE_CLASS_MAPPINGS = {
    "CheckpointPreviewLoader": CheckpointPreviewLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CheckpointPreviewLoader": "Checkpoint Preview Loader",
}