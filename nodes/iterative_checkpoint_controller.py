import os
from datetime import datetime
import folder_paths
from ..utils.kentskooking_utils import resolve_checkpoint_path, load_checkpoint_tensors


class IterativeCheckpointController:
    """
    Controls checkpoint configuration for iterative samplers (Image/Video).
    Supports creating new checkpointed runs or resuming from existing checkpoints.
    Validates checkpoint_type to ensure checkpoints are used with the correct sampler.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enable_checkpointing": ("BOOLEAN", {"default": True}),
                "checkpoint_interval": ("INT", {"default": 16, "min": 1, "max": 100, "step": 1}),
            },
            "optional": {
                "load_checkpoint": ("BOOLEAN", {"default": False, "label_on": "load"}),
                "checkpoint_run_id": ("STRING", {
                    "default": "",
                    "placeholder": "Run ID or checkpoint path"
                }),
            },
            "hidden": {
                # Backwards compatibility fallback for old workflows
                "load_video_checkpoint": "BOOLEAN",
            }
        }

    RETURN_TYPES = ("CHECKPOINT_CONFIG",)
    RETURN_NAMES = ("checkpoint_config",)
    FUNCTION = "generate_config"
    CATEGORY = "kentskooking/checkpointing"

    def generate_config(self, enable_checkpointing, checkpoint_interval, load_checkpoint=False,
                       checkpoint_run_id="", load_video_checkpoint=False):
        """Generate checkpoint configuration for the sampler."""

        # Backwards compatibility: use either load_checkpoint or load_video_checkpoint
        should_load = load_checkpoint or load_video_checkpoint

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
        if should_load:
            # Resume from existing checkpoint
            checkpoint_run_id = (checkpoint_run_id or "").strip()

            if not checkpoint_run_id:
                raise ValueError(
                    "You enabled load_checkpoint=True but no checkpoint_run_id was provided."
                )

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
        
        # Resolve full path using shared utility
        final_path, clean_run_id = resolve_checkpoint_path(run_id, checkpoint_dir)

        print(f"\n{'='*60}")
        print(f"Loading checkpoint: {clean_run_id}")
        print(f"{'='*60}")

        # Load tensors and metadata safely
        tensors, metadata = load_checkpoint_tensors(final_path)
        
        stacked_latents = tensors["latent_tensor"]
        loaded_current_latent = tensors.get("current_latent", None)

        # Get tensor count for fallback when metadata is missing
        latent_tensor_count = stacked_latents.shape[0]

        frame_count = int(metadata.get("frame_count", 0))
        last_frame_idx = int(metadata.get("last_frame_idx", frame_count - 1))
        checkpoint_type = metadata.get("checkpoint_type", "unknown")

        # Also check for iteration-based checkpoints (ImageIterativeSampler)
        iteration_count = int(metadata.get("iteration_count", 0))
        last_iteration_idx = int(metadata.get("last_iteration_idx", -1))
        total_iterations = int(metadata.get("total_iterations", 0))

        # Determine if this is a frame-based or iteration-based checkpoint
        if iteration_count > 0:
            # Iteration-based checkpoint (ImageIterativeSampler)
            loaded_latents = [stacked_latents[i:i+1].clone() for i in range(iteration_count)]
            resume_frame = last_iteration_idx + 1
            print(f"✓ Loaded {iteration_count} iterations from checkpoint")
            print(f"✓ Resuming from iteration {resume_frame}")
        elif frame_count > 0:
            # Frame-based checkpoint (VideoIterativeSamplerAdvanced)
            loaded_latents = [stacked_latents[i:i+1].clone() for i in range(frame_count)]
            resume_frame = last_frame_idx + 1
            print(f"✓ Loaded {frame_count} frames from checkpoint")
            print(f"✓ Resuming from frame {resume_frame}")
        else:
            # Fallback: metadata missing, infer from tensor shape
            loaded_latents = [stacked_latents[i:i+1].clone() for i in range(latent_tensor_count)]
            resume_frame = latent_tensor_count
            print(f"⚠ Metadata missing - inferring {latent_tensor_count} frames from tensor shape")
            print(f"✓ Resuming from frame {resume_frame}")

        print(f"✓ Checkpoint type: {checkpoint_type}")
        print(f"✓ File handle released (ready for new checkpoints)")
        print(f"{'='*60}\n")

        return ({
            "checkpoint_enabled": True,
            "checkpoint_interval": checkpoint_interval,
            "checkpoint_run_id": clean_run_id,
            "checkpoint_dir": checkpoint_dir,
            "resume_frame": resume_frame,
            "loaded_latents": loaded_latents,
            "loaded_current_latent": loaded_current_latent,
            "checkpoint_type": checkpoint_type,
            "total_iterations": total_iterations,
        },)


NODE_CLASS_MAPPINGS = {
    "IterativeCheckpointController": IterativeCheckpointController,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IterativeCheckpointController": "Iterative Checkpoint Controller",
}