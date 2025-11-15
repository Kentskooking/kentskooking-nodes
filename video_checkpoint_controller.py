import os
from datetime import datetime
import safetensors.torch
import folder_paths


class VideoCheckpointController:
    """
    Controls checkpoint configuration for VideoIterativeSamplerAdvanced.
    Supports creating new checkpointed runs or resuming from existing checkpoints.
    """

    @staticmethod
    def checkpoint_run_choices():
        """Return available checkpoint run IDs discovered on disk."""
        output_dir = folder_paths.get_output_directory()
        checkpoint_dir = os.path.join(output_dir, "video_checkpoints")
        if not os.path.isdir(checkpoint_dir):
            return []

        choices = []
        for entry in os.listdir(checkpoint_dir):
            if not entry.startswith("video_ckpt_") or not entry.endswith(".latent"):
                continue
            run_id = entry[len("video_ckpt_"):-len(".latent")]
            if run_id:
                choices.append(run_id)

        # Sort newest (based on timestamp naming) first so recent runs show up at the top.
        choices.sort(reverse=True)
        return choices

    @classmethod
    def INPUT_TYPES(cls):
        checkpoint_choices = cls.checkpoint_run_choices()
        checkpoint_input = (
            (checkpoint_choices, {
                "tooltip": "Select a previously saved video checkpoint run ID.",
                "default": checkpoint_choices[0] if checkpoint_choices else "",
            })
            if checkpoint_choices
            else ("STRING", {
                "default": "",
                "tooltip": "No checkpoints detected yet. Either run with checkpointing enabled or enter an ID manually.",
                "forceInput": True,
            })
        )

        return {
            "required": {
                "enable_checkpointing": ("BOOLEAN", {"default": True}),
                "checkpoint_interval": ("INT", {"default": 16, "min": 1, "max": 100, "step": 1}),
            },
            "optional": {
                "load_video_checkpoint": ("BOOLEAN", {"default": False}),
                "checkpoint_run_id": checkpoint_input,
            }
        }

    RETURN_TYPES = ("CHECKPOINT_CONFIG",)
    RETURN_NAMES = ("checkpoint_config",)
    FUNCTION = "generate_config"
    CATEGORY = "kentskooking/checkpointing"

    def generate_config(self, enable_checkpointing, checkpoint_interval, load_video_checkpoint=False,
                       checkpoint_run_id=""):
        """Generate checkpoint configuration for the sampler."""

        # Create checkpoint directory
        output_dir = folder_paths.get_output_directory()
        checkpoint_dir = os.path.join(output_dir, "video_checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        if not enable_checkpointing:
            # Return disabled config
            return ({
                "checkpoint_enabled": False,
                "checkpoint_interval": 0,
                "checkpoint_run_id": "",
                "checkpoint_dir": "",
                "resume_frame": 0,
                "loaded_latents": None,
            },)

        # Enabled checkpointing - either new run or resume
        if load_video_checkpoint:
            # Resume from existing checkpoint
            if not checkpoint_run_id:
                raise ValueError("checkpoint_run_id is required when load_video_checkpoint is True")

            return self._load_checkpoint(checkpoint_run_id, checkpoint_dir, checkpoint_interval)
        else:
            # New checkpointed run
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

            return ({
                "checkpoint_enabled": True,
                "checkpoint_interval": checkpoint_interval,
                "checkpoint_run_id": run_id,
                "checkpoint_dir": checkpoint_dir,
                "resume_frame": 0,
                "loaded_latents": None,
            },)

    def _load_checkpoint(self, run_id, checkpoint_dir, checkpoint_interval):
        """Load existing checkpoint and return config with resumed state."""
        # Strip "video_ckpt_" prefix if user included it
        if run_id.startswith("video_ckpt_"):
            run_id = run_id[len("video_ckpt_"):]

        # Strip ".latent" extension if user included it
        if run_id.endswith(".latent"):
            run_id = run_id[:-len(".latent")]

        checkpoint_path = os.path.join(checkpoint_dir, f"video_ckpt_{run_id}.latent")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        print(f"\n{'='*60}")
        print(f"Loading checkpoint: {run_id}")
        print(f"{'='*60}")

        # Load checkpoint file using context manager to ensure file is properly closed
        # This prevents Windows error 1224 when trying to save new checkpoints later
        import torch
        with safetensors.safe_open(checkpoint_path, framework="pt", device="cpu") as f:
            # Extract metadata
            metadata = f.metadata() or {}

            # Load tensor and clone it to ensure it's copied to memory (not memory-mapped)
            # Without .clone(), the tensor stays memory-mapped and locks the file
            stacked_latents = f.get_tensor("latent_tensor").clone()

        frame_count = int(metadata.get("frame_count", 0))
        last_frame_idx = int(metadata.get("last_frame_idx", frame_count - 1))

        # Split stacked latents back into list of individual frames
        loaded_latents = [stacked_latents[i:i+1].clone() for i in range(frame_count)]

        resume_frame = last_frame_idx + 1

        print(f"✓ Loaded {frame_count} frames from checkpoint")
        print(f"✓ Resuming from frame {resume_frame}")
        print(f"✓ File handle released (ready for new checkpoints)")
        print(f"{'='*60}\n")

        return ({
            "checkpoint_enabled": True,
            "checkpoint_interval": checkpoint_interval,
            "checkpoint_run_id": run_id,
            "checkpoint_dir": checkpoint_dir,
            "resume_frame": resume_frame,
            "loaded_latents": loaded_latents,
        },)


NODE_CLASS_MAPPINGS = {
    "VideoCheckpointController": VideoCheckpointController,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoCheckpointController": "Video Checkpoint Controller",
}
