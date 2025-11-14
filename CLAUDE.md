# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a ComfyUI custom node package that provides triangle wave-controlled iterative video sampling. It allows dynamic parameter modulation (steps, denoise, zoom, CLIP strength) across video frames using triangle wave patterns.

## External Nodes Used as Reference

This package references external ComfyUI nodes for implementation patterns and functionality:

1. **Comfyui-FeedbackSampler** (`../Comfyui-FeedbackSampler/`)
   - Template for frame-by-frame iteration pattern
   - ComfyUI integration structure reference

2. **comfyui-mixlab-nodes** (`../comfyui-mixlab-nodes/`)
   - **NoiseImage** node (`nodes/ImageNode.py:2808`)
   - Referenced for noise injection implementation
   - Generates white base image with uniform random noise per channel
   - Parameters: `noise_level=8192`, `color_hex=#FFFFFF`
   - Method: `create_noisy_image()` uses `random.randint(-noise_level, noise_level)` per RGB channel

3. **ComfyUI-KJNodes** (`../ComfyUI-KJNodes/`)
   - **InjectNoiseToLatent** node (`nodes/nodes.py:1039`)
   - Referenced for latent noise blending implementation
   - Method: `noised = samples + noise * strength`
   - Supports optional normalization, masking, and additional randn mixing

4. **comfyui_ipadapter_plus** (`../comfyui_ipadapter_plus/`)
   - **IPAdapterPlus** (`IPAdapterPlus.py`)
   - Used for per-frame IPAdapter application
   - Function: `ipadapter_execute()` applies image conditioning to models

5. **Inspire Pack** (`../comfyui-inspire-pack/`)
   - **KSamplerAdvanced** reference (`inspire/a1111_compat.py:64`)
   - Clarified advanced sampling parameter mapping
   - Method: `inspire_ksampler()` shows correct `steps`/`start_at_step`/`end_at_step` usage

## Node Architecture

The package consists of three interconnected nodes:

1. **TriangleWaveController** (`triangle_wave_controller.py`)
   - Generates configuration for triangle wave parameters
   - Outputs: `TRIANGLE_WAVE_CONFIG` (custom data type)
   - Contains the `triangle_wave()` function and parameter ranges (min/max for steps, denoise, zoom, clip_strength)

2. **VideoIterativeSampler** (`video_iterative_sampler.py`)
   - Main processing node that iterates through video latent batches frame-by-frame
   - Accepts optional `TRIANGLE_WAVE_CONFIG` to enable dynamic parameters
   - Each frame is processed individually through `nodes.common_ksampler()`
   - Supports:
     - Dynamic parameter calculation per frame
     - Style model application with variable strength
     - Noise injection for detail enhancement
     - Zoom/scale transformations on latents
     - Previous frame feedback mode
   - After processing, calls `comfy.model_management.unload_all_models()` and `soft_empty_cache()`

3. **WaveVisualizer** (`wave_visualizer.py`)
   - Debugging/preview node that visualizes wave patterns
   - Takes `TRIANGLE_WAVE_CONFIG` and generates a graph image
   - Returns IMAGE tensor compatible with ComfyUI's image system

## Data Flow

```
TriangleWaveController → TRIANGLE_WAVE_CONFIG → VideoIterativeSampler → LATENT batch
                                               ↘
                                          WaveVisualizer → IMAGE (preview)
```

## Key Implementation Details

### Triangle Wave Calculation
The triangle wave function (duplicated in both controller and sampler for independence):
- Creates smooth up-and-down oscillation within a cycle
- Position in cycle: `frame_idx % cycle_length`
- Linear interpolation from min to max in first half of cycle, max to min in second half

### Style Model Application
VideoIterativeSampler includes `apply_style_model()` method based on ComfyUI's StyleModelApply node (nodes.py:1051):
- Modifies conditioning with CLIP vision embeddings
- Supports two strength types: "multiply" and "attn_bias"
- Handles attention mask manipulation for proper token positioning

### Latent Zoom Transformation
The `zoom_latent()` method:
- `zoom > 1.0`: crops center and scales up (zoom in)
- `zoom < 1.0`: scales down and pads (zoom out)
- Uses bilinear interpolation via `F.interpolate()`

### Frame Processing
Each frame processes independently in sequence:
1. Calculate dynamic parameters from wave config (if enabled)
2. Apply style model to positive conditioning (if provided)
3. Apply previous frame feedback (if enabled)
4. Apply zoom transformation
5. Inject noise (if enabled)
6. Run through KSampler with calculated parameters
7. Store result and clone for potential feedback

## ComfyUI Integration

### Node Registration
Nodes are registered via `__init__.py` using the standard ComfyUI pattern:
- `NODE_CLASS_MAPPINGS`: maps internal class names to classes
- `NODE_DISPLAY_NAME_MAPPINGS`: maps internal names to UI display names

### Custom Data Type
`TRIANGLE_WAVE_CONFIG` is a dictionary containing:
- `cycle_length`, parameter min/max values
- The triangle_wave function reference (though VideoIterativeSampler re-implements it)

### ComfyUI API Dependencies
- `comfy.sample`, `comfy.samplers`: sampling system
- `comfy.model_management`: VRAM management
- `nodes.common_ksampler()`: core sampling function
- Standard ComfyUI types: MODEL, CONDITIONING, LATENT, STYLE_MODEL, CLIP_VISION_OUTPUT, IMAGE

## Development Notes

- No external package manager (npm/cargo/etc) - pure Python
- Testing should be done by loading nodes into ComfyUI and creating test workflows
- The package works standalone within ComfyUI's custom_nodes directory
- No build/compile step required - direct Python execution
- ComfyUI auto-discovers nodes via `__init__.py` exports
- When extending functionality, refer to `../Comfyui-FeedbackSampler/` for additional frame iteration patterns

## Dependencies

All core dependencies are provided by ComfyUI's main requirements.txt:
- `torch` - PyTorch for tensor operations
- `numpy>=1.25.0` - Numerical operations
- `Pillow` - Image processing (preview PNG generation)
- `safetensors>=0.4.2` - Checkpoint file format

Optional dependencies (other custom nodes):
- `comfyui_ipadapter_plus` - Required for WaveIPAdapter nodes (IPAdapter wave modulation)
  - Install: `cd custom_nodes && git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus.git comfyui_ipadapter_plus`
  - If not installed, IPAdapter nodes will be disabled gracefully

See `requirements.txt` for reference (no installation needed if ComfyUI is already set up).

## Session Log (2024-xx-xx)

- Added `WaveStyleModelApply` node to act as a CLIP StyleModelApply drop-in that also forwards `TRIANGLE_WAVE_CONFIG` metadata, enabling wave-config chaining through conditioning helpers.
- Refined `VideoIterativeSampler` inputs to rely solely on the wave-config payload for style data, keeping the sampler UI minimal while still modulating clip strength per frame.
- Expanded the example workflow (stored separately) to include all kentskooking nodes wired together for quick inspection.
- Updated `WaveVisualizer` to draw each parameter in its own horizontal band with min/max labels so overlapping waveforms are easier to read.
- Added `WaveClipVisionEncode`, a CLIP vision encoder that enriches the wave config with a sequence of embeddings for per-cycle selection.
- `WaveStyleModelApply` now accepts optional direct CLIP vision inputs but prefers the sequence stored on the wave config, enabling node chaining without extra sockets.
- `VideoIterativeSampler` pulls the appropriate CLIP embedding for each frame by wrapping through the stored sequence (`((frame_idx // cycle_length) % len(sequence))`), ensuring consistent looping even when frame counts exceed provided images.
- Introduced advanced counterparts:
  - `TriangleWaveControllerAdvanced` outputs step-floor/start/end metadata while still modulating zoom and CLIP strength.
  - `VideoIterativeSamplerAdvanced` implements advanced sampling with start_step/end_step oscillation per frame.

### Advanced Sampling Fix (2025-01-12) ✓ RESOLVED
**Problem**: Manual re-noising in `VideoIterativeSamplerAdvanced` caused blurry results. Initial approach tried to manually noise latents using `process_latent_in/out` + `model_sampling.noise_scaling` before calling `common_ksampler` with `disable_noise=True`.

**Root Cause**: ComfyUI's sampling pipeline already handles advanced sampling correctly. When `start_step` is provided, it:
- Slices the sigma schedule to `sigmas[start_step:]`
- Applies noise using the correct sigma (sigmas[0] after slicing = original sigmas[start_step])
- Manual pre-noising caused double-noising or incorrect sigma application

**Solution**:
- Removed all manual noise application code
- Pass clean (unnoised) latent to `common_ksampler`
- Use `disable_noise=(not add_noise)` to let ComfyUI handle noise internally
- Reference: Inspire Pack's `a1111_compat.py:inspire_ksampler()` uses the same approach

**Key Insight**: The `latent_image` parameter in ComfyUI's sampling functions expects the CLEAN latent. Noise is applied internally at the correct sigma level based on start_step/end_step parameters.

### Sigma Schedule Length Fix (2025-01-12) ✓ RESOLVED
**Problem**: `VideoIterativeSamplerAdvanced` produced completely different results compared to native KSamplerAdvanced with identical settings. Running steps 20-25 caused massive transformations instead of subtle refinements. Visual comparison showed severe distortion vs clean native results.

**Root Cause**: Incorrect parameter mapping to `common_ksampler`. The sampler was misunderstanding the relationship between the controller's parameters and the native KSamplerAdvanced parameters.

**How Native KSamplerAdvanced Works**:
Native KSamplerAdvanced has THREE parameters:
- `steps`: (e.g., 25) - defines total sigma schedule length
- `start_at_step`: (e.g., 20) - where to start in that schedule
- `end_at_step`: (default 10000) - fallback max, usually ignored when less than steps
- Example: steps=25, start=20, end=10000 → runs 5 actual steps (20→25) with proper sigma for "steps 20-25 of a 25-step schedule"

**Our Triangle Wave Controller Advanced**:
User sets:
- `start_at_step`: 20 (where to start)
- `end_at_step`: 40 (defines the total schedule length, NOT where to end)

**Correct Parameter Mapping**:
The controller's `end_at_step` is actually the total sigma schedule length:
- Controller's `end_at_step` (e.g., 40) → Sampler's `steps` parameter
- Controller's `start_at_step` (e.g., 20) → Sampler's `start_step` parameter
- Sampler's `last_step` → Always 10000 (fallback, effectively ignored)

Example workflow:
- User sets controller: start_at_step=20, end_at_step=40
- Wave oscillates end_at_step between 30-40 across frames
- Frame 1: sampler receives steps=30, start_step=20, last_step=10000 → runs 10 actual steps
- Frame 30: sampler receives steps=40, start_step=20, last_step=10000 → runs 20 actual steps
- All with proper sigma noise for their position in the schedule

**Solution**:
Changed `common_ksampler` call in `VideoIterativeSamplerAdvanced.process_video`:
```python
# Before (WRONG):
result = nodes.common_ksampler(
    frame_model, frame_seed, total_steps, cfg, sampler_name, scheduler,
    frame_positive, negative, latent_dict,
    disable_noise=(not add_noise), start_step=start_step, last_step=end_step, force_full_denoise=True
)

# After (CORRECT):
result = nodes.common_ksampler(
    frame_model, frame_seed, end_step, cfg, sampler_name, scheduler,
    frame_positive, negative, latent_dict,
    disable_noise=(not add_noise), start_step=start_step, last_step=10000, force_full_denoise=True
)
```

**Key Insight**: The controller's `end_at_step` parameter is actually defining the total sigma schedule length, not the ending position. This matches how users naturally think about sampling ranges while providing the correct sigma noise levels.

**Code Location**: `video_iterative_sampler_advanced.py:313`

### IPAdapter Integration (2025-01-12)
- Added `WaveIPAdapterController` node to modulate IPAdapter weight, start_at, and end_at parameters via triangle waves
- Added `WaveIPAdapterAdvanced` node as wrapper that stores IPAdapter config in wave_config for per-frame application
- Updated both `VideoIterativeSampler` and `VideoIterativeSamplerAdvanced` to:
  - Import `ipadapter_execute` from comfyui_ipadapter_plus
  - Clone model per-frame and apply IPAdapter with calculated wave values
  - Maintains single-run behavior (Option A approach) for clean, conflict-free operation
- Added image batch support: multiple images cycle through at each complete wave cycle
  - Uses same pattern as `WaveClipVisionEncode`
  - Calculation: `(frame_idx // cycle_length) % image_count`
- IPAdapter weight, start_at, end_at now oscillate smoothly across video frames
- Supports all native IPAdapter parameters: weight_type, combine_embeds, embeds_scaling, attn_mask, etc.

### Zoom Interpolation Update (2025-01-12)
- Changed zoom latent interpolation from `bilinear` to `bicubic` in both samplers
- Provides sharper, higher-quality zoom transformations
- Closer to Lanczos quality while remaining PyTorch-native

### WaveVisualizer Enhancement (2025-01-12)
- Added support for both basic and advanced triangle wave controllers
- Automatically detects `controller_variant == "advanced"`
- Advanced mode displays: Start Step, End Step, Steps (derived), Zoom, CLIP Strength
- Basic mode displays: Steps, Denoise, Zoom, CLIP Strength

### VAE-Encoded Noise Injection (2025-01-13)
**Problem**: Original noise injection used pure Gaussian noise (`torch.randn_like()`) which doesn't respect VAE latent space structure, potentially causing artifacts.

**User's External Workflow**:
1. Generate noise image with Mixlab NoiseImage (white base + uniform random noise, level=8192)
2. VAE encode the noise image → structured latent noise
3. Blend with KJ-Nodes InjectNoiseToLatent: `latent + noise_latent * 0.1`

**Implementation**:
Added proper VAE-encoded noise injection to both samplers:

```python
def generate_noise_image(self, width, height, seed, noise_level=8192):
    """Generate noise image matching Mixlab NoiseImage node."""
    random.seed(seed)
    image = Image.new("RGB", (width, height), (255, 255, 255))
    pixels = image.load()
    for i in range(width):
        for j in range(height):
            # Uniform distribution noise per channel (NOT Gaussian)
            noise_r = random.randint(-noise_level, noise_level)
            noise_g = random.randint(-noise_level, noise_level)
            noise_b = random.randint(-noise_level, noise_level)
            # Clamp to 0-255
            r = max(0, min(pixels[i, j][0] + noise_r, 255))
            g = max(0, min(pixels[i, j][1] + noise_g, 255))
            b = max(0, min(pixels[i, j][2] + noise_b, 255))
            pixels[i, j] = (r, g, b)
    return image

def inject_noise_latent(self, latent, strength, vae=None, seed=0):
    if vae is not None:
        # Generate noise in pixel space, VAE encode, then blend
        noise_image = self.generate_noise_image(pixel_width, pixel_height, seed, noise_level=8192)
        noise_tensor = self.pil_to_tensor(noise_image)
        noise_latent = vae.encode(noise_tensor[:,:,:,:3])
        return latent + noise_latent * strength
    else:
        # Fallback: simple Gaussian noise
        return latent + torch.randn_like(latent) * strength
```

**Changes**:
- Added optional `vae` input to both samplers
- Changed `noise_injection_strength` default from 0.0 to 0.1
- Split seed handling for proper video consistency:
  - `noise_seed = seed + frame_idx` → Varies per frame for noise injection
  - `sampler_seed = seed` → **CONSTANT across all frames** for diffusion sampling
- When VAE provided: Uses VAE-encoded noise (superior quality, respects latent space)
- When VAE not provided: Falls back to Gaussian noise (maintains backward compatibility)

**Benefits**:
- VAE-aware noise maintains image coherence
- Prevents artifacts from raw random values in latent space
- Matches proven external workflow
- User-controllable blend ratio via `noise_injection_strength`

**CRITICAL: Seed Management for Video Consistency**:
⚠️ **DO NOT CHANGE**: The sampler seed (`sampler_seed = seed`) MUST remain constant across all frames for temporal consistency in video generation. Only the noise injection seed (`noise_seed = seed + frame_idx`) should vary per frame. Changing the sampler seed per frame will break video coherence and cause inconsistent results between frames. The sampler uses the same underlying seed with noise patterns adjusted only by the number of steps, NOT by frame number.

**Code Locations**:
- `video_iterative_sampler.py:152-216, 371-390`
- `video_iterative_sampler_advanced.py:154-218, 301-382`

### ControlNet Integration (2025-01-13) ✓ COMPLETE
**Goal**: Add ControlNet integration with triangle wave modulation, similar to existing Style Model and IPAdapter implementations.

**Implementation**: Created 2 new nodes following the modular wave-config approach:

1. **WaveControlNetApply** (`wave_controlnet_apply.py`)
   - Stores controlnet data in wave_config, passthrough conditioning
   - Inputs: positive, negative, control_net, image, wave_config, vae (optional)
   - Outputs: positive, negative, wave_config (enriched)
   - Stores in wave_config:
     - `controlnet_model`: The ControlNet model
     - `controlnet_hint`: Preprocessed image (`image.movedim(-1, 1)`)
     - `controlnet_vae`: Optional VAE for control hint encoding

2. **WaveControlNetController** (`wave_controlnet_controller.py`)
   - Adds wave modulation parameters to wave_config
   - Parameters (all with _min/_max for triangle wave):
     - `strength_min/max`: ControlNet strength (0.0-10.0, default 0.5-1.0)
     - `start_at_min/max`: When to start applying (0.0-1.0, default 0.0-0.0)
     - `end_at_min/max`: When to stop applying (0.0-1.0, default 1.0-1.0)
   - Stores in wave_config: `controlnet_strength_min/max`, `controlnet_start_at_min/max`, `controlnet_end_at_min/max`
   - Includes `calculate_for_frame()` method for per-frame value calculation

3. **Sampler Integration** (both samplers)
   - Added `apply_controlnet()` method (based on nodes.py:874 ControlNetApplyAdvanced)
   - Added `calculate_controlnet_wave_values()` method
   - Per-frame application to both positive and negative conditioning
   - Application order: Style Model → ControlNet → IPAdapter → KSampler

**Issues Resolved**:
1. ✓ Workflow wiring - wave_config not passing through (user needed to connect WaveControlNetApply output)
2. ✓ Batch handling - Only first frame being used from control video batch
   - **Problem**: WaveControlNetApply stored entire control_hint batch, but samplers passed full batch to every frame
   - **Solution**: Extract correct frame per frame_idx: `frame_control_hint = control_hint_batch[frame_idx % batch_size:frame_idx % batch_size + 1]`
   - **Code locations**:
     - `video_iterative_sampler_advanced.py:401-405`
     - `video_iterative_sampler.py:412-416`

**Code Locations**:
- `wave_controlnet_apply.py:1-42` (full file)
- `wave_controlnet_controller.py:1-81` (full file)
- `__init__.py:10-11, 23-24, 37-38` (registration)
- `video_iterative_sampler.py:100-130, 184-205, 363-372, 401-432, 473` (apply method, calculation method, config extraction, per-frame application with batch slicing, use frame_negative)
- `video_iterative_sampler_advanced.py:95-125, 186-207, 347-356, 388-416, 465` (apply method, calculation method, config extraction, per-frame application with batch slicing, use frame_negative)

### IPAdapter Debugging (2025-01-13) ⚠️ TEMPORARY - TO BE REMOVED

**Issue**: IPAdapter wave control not working, needs diagnosis.

**⚠️ TEMPORARY DEBUG LOGGING** (Added 2025-01-13 - TO BE REMOVED):

Added verbose logging to diagnose IPAdapter data flow at 3 locations:

1. **wave_ipadapter_advanced.py:70-80** (in `store_config()` method):
```python
print(f"\n{'='*60}")
print(f"WaveIPAdapterAdvanced: Attaching IPAdapter to wave_config")
print(f"  IPAdapter model: {ipadapter}")
print(f"  Image batch shape: {image.shape}")
print(f"  Image sequence length: {len(image_sequence)}")
print(f"  Weight: {weight}, Weight type: {weight_type}")
print(f"  Start at: {start_at}, End at: {end_at}")
print(f"  CLIP vision: {clip_vision}")
print(f"  ipadapter_enabled: {enriched_config.get('ipadapter_enabled')}")
print(f"  Keys in wave_config: {list(enriched_config.keys())}")
print(f"{'='*60}\n")
```

2. **video_iterative_sampler_advanced.py:331-350** (IPAdapter detection):
```python
ipadapter_enabled = wave_config.get("ipadapter_enabled", False) and IPADAPTER_AVAILABLE
print(f"\n  IPAdapter detection: enabled={ipadapter_enabled}")
print(f"  ipadapter_enabled in wave_config: {wave_config.get('ipadapter_enabled', False)}")
print(f"  IPADAPTER_AVAILABLE: {IPADAPTER_AVAILABLE}")
if wave_config:
    print(f"  Keys in wave_config: {list(wave_config.keys())}")
if ipadapter_enabled:
    # ... extract config ...
    print(f"  IPAdapter config extracted successfully")
```

3. **video_iterative_sampler_advanced.py:444-446** (per-frame application):
```python
print(f"  Selecting IPAdapter image {cycle_index+1}/{seq_len} for frame {frame_idx+1}")
print(f"  Applying IPAdapter: weight={ipadapter_weight:.3f}, start={ipadapter_start_at:.3f}, end={ipadapter_end_at:.3f}")
```

4. **video_iterative_sampler.py:347-366** (IPAdapter detection - same as advanced)

5. **video_iterative_sampler.py:455-457** (per-frame application - same as advanced)

**To Remove Debug Logging**: Once issue is resolved, DELETE these specific print statements at the locations above.

### VideoIterativeSamplerAdvanced Simplification (2025-01-13)
- Advanced sampler now **requires** `TriangleWaveControllerAdvanced` config input (no more optional `wave_config`)
- Removed legacy fallback inputs (`base_start_at`, `base_end_at`, `use_dynamic_params`) and always derive steps/start/end directly from advanced wave metadata
- `calculate_wave_values()` now assumes advanced controller fields and no longer supports basic `steps_min/steps_max` configs
- Per-frame processing always references the wave config for zoom/clip sequencing; cycle detection simplified
- Runtime signature change documented in `video_iterative_sampler_advanced.py:24-120, 292-440`

### Frontend Node Width Standardization (2025-01-13)
- Added `js/kentskooking_node_width.js` extension that hooks into ComfyUI's client and forces every kentskooking node to share the same width
- Reference width is sampled from `TriangleWaveControllerAdvanced` (fallback 360px) so all nodes line up visually in workflows
- `__init__.py` now exports `WEB_DIRECTORY="./js"` so ComfyUI loads the extension automatically

### Noise Injection Seed Lock (2025-11-14)
**Feature**: Added `lock_injection_seed` boolean parameter to `VideoIterativeSamplerAdvanced` for controlling noise injection seed behavior.

**Implementation**:
- New UI parameter: `lock_injection_seed` (default: `False`, labels: "locked"/"varying")
- Positioned directly under `noise_injection_strength` in the UI
- Behavior:
  - `lock_injection_seed=False` (default): `noise_seed = seed + frame_idx` (varies per frame - original behavior)
  - `lock_injection_seed=True` + `noise_injection_strength > 0`: `noise_seed = seed + 1` (constant across all frames)
  - When `noise_injection_strength <= 0`: parameter is ignored (no injection occurs)

**Use Case**: Enables consistent noise patterns across all frames when locked, useful for maintaining uniform texture detail without per-frame variation. Unlocked mode provides temporal variation in noise injection.

**Code Location**: `video_iterative_sampler_advanced.py:52, 296, 372-379`

### Video Checkpoint System (2025-11-14)
**Feature**: Implemented checkpoint/resume capability for long video processing runs to recover from interruptions.

**Architecture**:
- **New Node**: `VideoCheckpointController` (config-only output, like wave controllers)
- **Modified Node**: `VideoIterativeSamplerAdvanced` (added optional `checkpoint_config` input)
- **Checkpoint Directory**: `{output_directory}/video_checkpoints/`
- **File Naming**:
  - Run ID format: `YYYYMMDD_HHMMSS` (auto-generated timestamp)
  - Preview PNG: `video_ckpt_{run_id}_preview.png` (deleted on successful completion)
  - Checkpoint file: `video_ckpt_{run_id}.latent` (deleted on successful completion)

**VideoCheckpointController Node** (Config-Only):
- Inputs:
  - `enable_checkpointing`: Boolean (default: **True**)
  - `checkpoint_interval`: Int (default: 16 frames)
  - `load_video_checkpoint`: Boolean (default: False) - for resuming
  - `checkpoint_run_id`: String - required when loading
- Outputs: `CHECKPOINT_CONFIG` (custom type containing checkpoint parameters)
- Behavior:
  - **Pure config generator** - no data inputs, no file I/O (follows wave controller pattern)
  - **New run**: Auto-generates run_id timestamp, returns config for new checkpoint
  - **Resume run**: Loads existing checkpoint file, extracts processed latents and resume frame index
  - Returns disabled config if `enable_checkpointing=False`

**VideoIterativeSamplerAdvanced Integration**:
- New optional input: `checkpoint_config` (CHECKPOINT_CONFIG type)
- **Ignores checkpoint operations if no config provided** (backward compatible)
- New methods:
  - `generate_checkpoint_preview()`: Creates preview PNG from first input latent frame
  - `save_checkpoint()`: Saves all processed latents to `.latent` file (overwrites previous)
  - `cleanup_checkpoint()`: Deletes both checkpoint .latent and preview PNG on successful completion
- Modified `process_video()`:
  - Extracts checkpoint parameters from config
  - Generates preview PNG for new runs (if VAE provided and checkpointing enabled)
  - Uses loaded latents if resuming (skips already-processed frames)
  - Starts loop from `resume_frame` instead of 0
  - Saves checkpoint every N frames (rolling update, same filename)
  - Auto-cleanup on successful completion (keeps preview PNG only)

**File Format**:
- Uses standard ComfyUI safetensors format (`.latent` extension)
- Structure: `{"latent_tensor": stacked_tensor, "latent_format_version_0": torch.tensor([])}`
- Metadata: `{"frame_count": str, "last_frame_idx": str, "checkpoint_type": str}`
- Compatible with ComfyUI's LoadLatent node

**Workflow Usage**:

*New checkpointed run*:
```
VideoCheckpointController (enable_checkpointing=True, checkpoint_interval=16)
  → CHECKPOINT_CONFIG → VideoIterativeSamplerAdvanced (vae=VAE for preview PNG)
```

*Resume from interruption*:
```
VideoCheckpointController (load_video_checkpoint=True, checkpoint_run_id="20251114_153045")
  → CHECKPOINT_CONFIG → VideoIterativeSamplerAdvanced
```

**Notes**:
- Preview PNG generation happens in the sampler (uses existing VAE input), not in the controller
- Both checkpoint files (.latent and preview PNG) are deleted on successful completion to avoid confusion
- Preview PNG contains full workflow metadata for deterministic resume

**Benefits**:
- ✅ Recover from crashes/interruptions without losing hours of work
- ✅ Rolling checkpoint (only one .latent per run, overwrites previous)
- ✅ Auto-cleanup (successful runs leave no checkpoint files)
- ✅ Minimal disk usage (~500KB PNG + variable .latent during run, all cleaned up after)
- ✅ Resume capability (load checkpoint_run_id to continue from interruption)
- ✅ Workflow metadata in preview PNG ensures deterministic resume

**Code Locations**:
- `video_checkpoint_controller.py:1-126` (full file, config-only implementation)
- `video_iterative_sampler_advanced.py:331-348` (cleanup_checkpoint - deletes both .latent and preview PNG)
- `video_iterative_sampler_advanced.py:350-392` (generate_checkpoint_preview with workflow metadata)
- `__init__.py:12, 26, 41` (node registration)

**Design Pattern**: Controller follows the same pure-config pattern as wave controllers (TriangleWaveController, WaveIPAdapterController, etc.) - no data processing, only configuration generation.

### Checkpoint Preview PNG - Workflow Metadata (2025-11-14) ✓ RESOLVED

**Critical Requirement**: The preview PNG must contain ComfyUI workflow metadata so users can drag-and-drop it to restore all initial conditions (seed, input images, etc.) for deterministic resume capability.

**Solution Implemented**: Added hidden inputs to VideoIterativeSamplerAdvanced to receive workflow metadata from ComfyUI's execution system.

**Implementation**:

1. **Added Hidden Inputs** (`video_iterative_sampler_advanced.py:59-62`):
```python
"hidden": {
    "prompt": "PROMPT",
    "extra_pnginfo": "EXTRA_PNGINFO"
}
```
These special input types are automatically populated by ComfyUI with workflow data (reference: `nodes.py:1575-1577` SaveImage node).

2. **Updated Method Signatures**:
- `process_video()` now accepts `prompt=None, extra_pnginfo=None` parameters
- `generate_checkpoint_preview()` now accepts and embeds workflow metadata

3. **Workflow Metadata Embedding** (`video_iterative_sampler_advanced.py:374-381`):
```python
# Add ComfyUI workflow metadata (for drag-and-drop restore)
if prompt is not None:
    metadata.add_text("prompt", json.dumps(prompt))
if extra_pnginfo is not None:
    for key in extra_pnginfo:
        metadata.add_text(key, json.dumps(extra_pnginfo[key]))
```

**Result**: Preview PNGs now contain full workflow metadata. Users can drag-and-drop the PNG into ComfyUI to perfectly restore:
- All node parameters (seed, CFG, steps, etc.)
- Input images and their sources
- Wave controller settings
- All connections and workflow structure
This ensures deterministic resume capability - continuing from checkpoint produces identical results as if never interrupted.

**Key Implementation Detail**: Image conversion follows ComfyUI's SaveImage pattern exactly. VAE decode returns images in `[batch, height, width, channels]` format - NO transpose needed. Manual transposing breaks execution.

**Code Locations**:
- `video_iterative_sampler_advanced.py:59-62` (hidden inputs)
- `video_iterative_sampler_advanced.py:343-385` (generate_checkpoint_preview with workflow metadata)
- `video_iterative_sampler_advanced.py:387-389` (process_video signature)
- `video_iterative_sampler_advanced.py:416-417` (preview generation call)

### Checkpoint File Locking Fix (2025-11-14) ✓ RESOLVED

**Problem**: When resuming from a checkpoint, new checkpoints couldn't be saved. Error: "The requested operation cannot be performed on a file with a user-mapped section open. (os error 1224)"

**Root Cause**: Windows-specific file locking issue. The checkpoint file was being loaded with `safetensors.torch.load_file()` which memory-maps the file and keeps it open. When the sampler tried to save a new checkpoint to the same path, Windows wouldn't allow overwriting a file that's currently in use.

**Solution**: Changed checkpoint loading to use context manager with explicit tensor cloning:

```python
# OLD (keeps file memory-mapped):
latent_data = safetensors.torch.load_file(checkpoint_path, device="cpu")
stacked_latents = latent_data["latent_tensor"]

# NEW (properly closes file handle):
with safetensors.safe_open(checkpoint_path, framework="pt", device="cpu") as f:
    metadata = f.metadata() or {}
    stacked_latents = f.get_tensor("latent_tensor").clone()
# File automatically closed when exiting context manager

# Also clone individual frame splits:
loaded_latents = [stacked_latents[i:i+1].clone() for i in range(frame_count)]
```

**Key Points**:
1. Use `safetensors.safe_open()` context manager instead of `load_file()`
2. Call `.clone()` on tensors to copy data to memory (breaks memory-mapping link)
3. File handle is automatically released when exiting `with` block
4. Now safe to overwrite the same checkpoint file during resumed runs

**Code Location**: `video_checkpoint_controller.py:90-105`

### Todo
- _None currently._
- Always check comfyui core nodes before implementing new processes. Many times it is already built in and automatic allowing us to piggyback on their streamlined implementation. Example: Their save image node logic handles transpose automatically so manually transposing breaks execution