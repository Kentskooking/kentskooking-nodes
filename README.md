# kentskooking-nodes

Wave-controlled iterative video processing for ComfyUI. Alters videos frame by frame using image diffusion pipelines. work in progress

## Overview

This node pack fundamentally creates a frame by frame vid2vid workflow that uses image diffusion to add progressive diffusion over each frame. TLDR: makes trippy videos reminiscent of deforum

I made this in an attempt to speed up and make my workflow more efficient. I had it working with standard nodes in comfyui but it took forever and couldn't be fully automated (that workflow provided in examples folder along with this nodepack's wf). Used most of the internal logic from the **[Comfyui-FeedbackSampler](https://github.com/pizurny/Comfyui-FeedbackSampler)** by pizurny

## **Example**

![Demo GIF](examples/assets/demo.gif)



**Key Features:**
-  **Wave/Cycle Control**: Smooth oscillation with configurable cycles for many parameters
-  **Frame-by-Frame Processing**: All video frames get batch processed efficiently within the ksampler
-  **Checkpoint System**: Resume interrupted renders without losing progress (especially useful on huge batches)
-  **Multiple Integrations**: FLUX Kontext/clipvision/style model, IPAdapter, ControlNet - all with wave modulation built in

## Installation

Git Clone 

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Kentskooking/kentskooking-nodes.git
```

Restart ComfyUI and the nodes will appear under the "kentskooking" category.

### Dependencies

All core dependencies are already provided by ComfyUI. No additional installation required.

**Optional** (to use IPAdapter wave nodes you'll need to install IPAdapter):
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus.git comfyui_ipadapter_plus

or use manager to install ipadapter
```

**I suggest just loading the workflow  in the examples folder. Currently just have a workflow setup for FLUX. I plan to add SDXL and SD1.5 workflows also but you can just build them yourself easily.**


## Node Reference

### Core Nodes

#### TriangleWaveController 
Basic wave controller for steps, denoise, zoom, and CLIP strength modulation.

**Outputs:** `TRIANGLE_WAVE_CONFIG`

**Parameters:**
- `cycle_length`: Number of frames per complete wave cycle
- `steps_min/max`: Range for sampling steps
- `denoise_min/max`: Range for denoise strength
- `zoom_min/max`: Range for zoom factor (>1.0 = zoom in, <1.0 = zoom out)
- `clip_strength_min/max`: Range for CLIP conditioning strength

#### TriangleWaveControllerAdvanced
Advanced controller for start_step/end_step based sampling.

**Outputs:** `TRIANGLE_WAVE_CONFIG`

**Parameters:**
- `cycle_length`: Number of frames per complete wave cycle
- `step_floor`: Fewest denoise steps that will be taken
- `start_at_step`: Fixed starting step for all frames
- `end_at_step`: Sets the max step level that will reset the cycle
- `zoom_min/max`: Range for zoom factor
- `clip_strength_min/max`: Range for CLIP conditioning strength (includes flux redux strength)

#### VideoIterativeSampler
Basic frame-by-frame video sampler with triangle wave support.

**Inputs:**
- `model`: MODEL
- `positive/negative`: CONDITIONING
- `latent_batch`: LATENT (video frames)
- `wave_config`: TRIANGLE_WAVE_CONFIG (optional)
- `vae`: VAE (optional, for noise injection)
- `seed`: Random seed
- `sampler_name/scheduler`: Sampling settings
- `cfg`: CFG scale
- `noise_injection_strength`: Level of influence injected noise has (0.00-1.00)
- `feedback_mode`: "none" or "previous_frame" (when enabled blends previous latent with current latent at 10%, exactly like the feedback sampler from 

**Outputs:** `LATENT` (processed video frames)

#### VideoIterativeSamplerAdvanced
Advanced sampler with start_step/end_step control and checkpointing.

**Inputs:** (same as basic sampler, plus:)
- `checkpoint_config`: CHECKPOINT_CONFIG (optional)
- `lock_injection_seed`: Lock noise seed across frames (boolean)

**Outputs:** `LATENT` (processed video frames)

### Wave Integration Nodes

#### WaveStyleModelApply
Apply style model with wave-modulated CLIP strength.

**Inputs:**
- `positive/negative`: CONDITIONING
- `style_model`: STYLE_MODEL
- `clip_vision_output`: CLIP_VISION_OUTPUT
- `wave_config`: TRIANGLE_WAVE_CONFIG

**Outputs:** `positive`, `negative`, `wave_config` (enriched)

#### WaveClipVisionEncode
Encode multiple images for per-cycle CLIP vision selection.

**Inputs:**
- `clip_vision`: CLIP_VISION
- `image`: IMAGE (batch)
- `wave_config`: TRIANGLE_WAVE_CONFIG

**Outputs:** `wave_config` (enriched with image sequence)

#### WaveIPAdapterController
Control IPAdapter wave modulation parameters.

**Parameters:**
- `weight_min/max`: IPAdapter weight range
- `start_at_min/max`: Start percentage range
- `end_at_min/max`: End percentage range

**Outputs:** `TRIANGLE_WAVE_CONFIG` (enriched)

#### WaveIPAdapterAdvanced
Apply IPAdapter with config stored in wave_config.

**Inputs:**
- `model`: MODEL
- `ipadapter`: IPADAPTER
- `image`: IMAGE (batch, cycles through per wave cycle)
- `wave_config`: TRIANGLE_WAVE_CONFIG
- Additional IPAdapter parameters...

**Outputs:** `model`, `wave_config` (enriched)

#### WaveControlNetApply
Store ControlNet data in wave_config for per-frame application.

**Inputs:**
- `positive/negative`: CONDITIONING
- `control_net`: CONTROL_NET
- `image`: IMAGE (control hint batch)
- `wave_config`: TRIANGLE_WAVE_CONFIG
- `vae`: VAE (optional)

**Outputs:** `positive`, `negative`, `wave_config` (enriched)

#### WaveControlNetController
Control ControlNet wave modulation parameters.

**Parameters:**
- `strength_min/max`: ControlNet strength range
- `start_at_min/max`: Start percentage range
- `end_at_min/max`: End percentage range

**Outputs:** `TRIANGLE_WAVE_CONFIG` (enriched)

### Utility Nodes

#### WaveVisualizer (Development)
Visualize triangle wave patterns over time.

**Inputs:**
- `wave_config`: TRIANGLE_WAVE_CONFIG
- `preview_frames`: Number of frames to preview (default: 100)

**Outputs:** `IMAGE` (graph visualization)

#### VideoCheckpointController
Configure checkpoint/resume system for long renders.

**Inputs:**
- `enable_checkpointing`: Enable checkpoint saves (boolean)
- `checkpoint_interval`: Save checkpoint every N frames
- `load_video_checkpoint`: Resume from existing checkpoint (boolean)
- `checkpoint_run_id`: Checkpoint ID to resume (format: YYYYMMDD_HHMMSS)

**Outputs:** `CHECKPOINT_CONFIG`

**Files Created:**
- `{ComfyUI_output}/video_checkpoints/video_ckpt_{run_id}.latent` This is the checkpoint file (deleted on completion)
- `{ComfyUI_output}/video_checkpoints/video_ckpt_{run_id}_preview.png` This image contains the workflow (deleted on completion)

The preview PNG contains full workflow metadata, drop it into comfy and you can resume with all the original parameters

## Usage Examples

### Basic Wave-Controlled Video

```
TriangleWaveController
  (cycle_length=20, steps_min=20, steps_max=40)
    ↓
VideoIterativeSampler
  (model, positive, negative, latent_batch, wave_config)
    ↓
VAEDecode
    ↓
SaveVideo
```

### Advanced with Checkpointing

```
TriangleWaveControllerAdvanced
  (cycle_length=20, start_at_step=16, end_at_step_min=23, end_at_step_max=36)
    ↓
VideoCheckpointController
  (enable_checkpointing=True, checkpoint_interval=16)
    ↓
VideoIterativeSamplerAdvanced
  (wave_config, checkpoint_config, vae)
    ↓
VAEDecode
```

### With Style Model + IPAdapter

```
LoadImage (style reference)
    ↓
CLIPVisionEncode
    ↓
TriangleWaveController
    ↓
WaveStyleModelApply
  (style_model, clip_vision_output, wave_config)
    ↓
WaveIPAdapterController
    ↓
WaveIPAdapterAdvanced
  (ipadapter, image_batch, wave_config)
    ↓
VideoIterativeSamplerAdvanced
```

## How Triangle Waves Work

Triangle waves create smooth up-and-down oscillations:

```
Max ────┐      ┌────┐
        │     /│\   │
        │    / │ \  │
        │   /  │  \ │
        │  /   │   \│
Min ────┴─     └────┘
        └─ cycle_length ─┘
```

**Frame calculation:**
- Position in cycle: `frame_idx % cycle_length`
- First half: Linear interpolation from min → max
- Second half: Linear interpolation from max → min

## Checkpoint System

### Creating a Checkpointed Run

1. Add `VideoCheckpointController` with `enable_checkpointing=True`
2. Connect to `VideoIterativeSamplerAdvanced`
3. Connect a VAE (generates workflow metadata PNG)
4. Run your workflow

**Files created during run:**
- `video_ckpt_{timestamp}.latent` - Rolling checkpoint (overwrites each interval)
- `video_ckpt_{timestamp}_preview.png` - Workflow metadata (for resuming)

### Resuming from Interruption

1. Set `VideoCheckpointController`:
   - `load_video_checkpoint=True`
   - `checkpoint_run_id="YYYYMMDD_HHMMSS"` (from filename)
2. Run workflow - continues from last checkpoint

**OR drag-and-drop the preview PNG** to restore the exact workflow, then enable checkpoint loading.

### Completion

Both files are **automatically deleted** when the video completes successfully.

## Advanced Features

### VAE-Encoded Noise Injection

Generates noise in pixel space, VAE-encodes it, then blends with input latent (modeled after Mixlab's NoiseImage node):

```python
noise_latent = vae.encode(noise_image)
result = latent + noise_latent * strength
```

Benefits:
- Respects VAE latent space structure
- Prevents artifacts from raw random noise
- User-controllable blend ratio

### Feedback Mode

`feedback_mode="previous_frame"` blends each frame's output with the next frame's input, creating temporal coherence.

### Seed Management

- Sampler seed: **CONSTANT** across all frames for temporal coherence
 -Even when set to randomize, increment, decrement, the same seed is used across all frames
- Noise injection seed: The noise injection seed varies from the main generation seed, and by default it changes each iteration. You can lock this unique seed using the "LOCK" option in the sampler for smoother animations.

## Support

- **Issues:** https://github.com/Kentskooking/kentskooking-nodes/issues
- **Discussions:** https://github.com/Kentskooking/kentskooking-nodes/discussions

## Attribution

This project was inspired by and references code from the following excellent ComfyUI custom nodes:

- **[Comfyui-FeedbackSampler](https://github.com/pizurny/Comfyui-FeedbackSampler)** by pizurny
  - Frame-by-frame iteration pattern and structure

- **[comfyui-mixlab-nodes](https://github.com/MixLabPro/comfyui-mixlab-nodes)** by MixLabPro
  - NoiseImage node implementation for VAE-encoded noise injection

- **[ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes)** by kijai
  - InjectNoiseToLatent blending approach

- **[ComfyUI_IPAdapter_plus](https://github.com/cubiq/ComfyUI_IPAdapter_plus)** by cubiq
  - IPAdapter integration and execution logic

- **[ComfyUI Inspire Pack](https://github.com/ltdrdata/ComfyUI-Inspire-Pack)** by ltdrdata
  - KSamplerAdvanced parameter mapping reference