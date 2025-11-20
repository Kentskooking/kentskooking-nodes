import torch
import safetensors.torch
import os


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

        # Check if user provided a directory path or just a filename/run_id
        directory = os.path.dirname(checkpoint_path)
        filename = os.path.basename(checkpoint_path)

        # Default to ./output/video_checkpoints/ if no directory specified
        if not directory:
            import folder_paths
            output_dir = folder_paths.get_output_directory()
            directory = os.path.join(output_dir, "video_checkpoints")

        # Strip known prefixes if present
        if filename.startswith("video_ckpt_"):
            filename = filename[len("video_ckpt_"):]
        elif filename.startswith("image_ckpt_"):
            filename = filename[len("image_ckpt_"):]

        # Strip ".latent" extension if present
        if filename.endswith(".latent"):
            filename = filename[:-len(".latent")]

        # Try to find the file with either prefix
        video_path = os.path.join(directory, f"video_ckpt_{filename}.latent")
        image_path = os.path.join(directory, f"image_ckpt_{filename}.latent")

        if os.path.exists(video_path):
            final_path = video_path
        elif os.path.exists(image_path):
            final_path = image_path
        else:
            raise FileNotFoundError(
                f"CheckpointPreviewLoader: Checkpoint file not found.\n"
                f"Tried: {video_path}\n"
                f"  and: {image_path}"
            )

        # Load .latent file using safetensors (same format as VideoCheckpointController saves)
        with safetensors.safe_open(final_path, framework="pt", device="cpu") as f:
            tensors = list(f.keys())

            if "latent_tensor" not in tensors:
                raise Exception(f"CheckpointPreviewLoader: 'latent_tensor' key not found in checkpoint. Available keys: {tensors}")

            samples = f.get_tensor("latent_tensor").clone()

        # Wrap in LATENT dict for ComfyUI
        return ({"samples": samples},)


NODE_CLASS_MAPPINGS = {
    "CheckpointPreviewLoader": CheckpointPreviewLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CheckpointPreviewLoader": "Checkpoint Preview Loader",
}
