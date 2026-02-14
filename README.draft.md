# kentskooking-nodes

Wave-driven video and image iteration tools for ComfyUI.
If you like animated diffusion workflows (Deforum-style motion, morphs, and prompt travel), this pack is built for that.

## What This Pack Does

- Runs frame-by-frame (or iteration-by-iteration) sampling with wave controls.
- Modulates things like steps, denoise, zoom, CLIP strength, ControlNet, and IPAdapter over time.
- Supports long runs with checkpoint + resume.
- Includes an Explorer sampler for prompt/conditioning traversals (A->B, A->B->C, optional loop back to A).

## Quick Start

1. Clone into `ComfyUI/custom_nodes`
2. Restart ComfyUI
3. Load one of the included workflows in `examples/workflows/`

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Kentskooking/kentskooking-nodes.git
```

Optional dependency (only for IPAdapter nodes):

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus.git comfyui_ipadapter_plus
```

## Included Example Workflows

- `examples/workflows/FLUX Video Iterative WF_kentskooking-nodes.json`
- `examples/workflows/Video Interpolate WF - FLUX.json`
- `examples/workflows/Video Interpolate WF (SDXL).json`
- `examples/workflows/Video Interpolate WF (SD 1.5).json`
- `examples/workflows/kentskooking_nodes_all.json`
- `examples/workflows/WF_using_standard_nodes.json` (reference baseline)

## Main Nodes (Simple Map)

### Controllers

- `Video Wave Controller`
- `Image Wave Controller`
- `Interpolation Wave Controller`
- `Explorer Conditioning Wave Controller`

### Samplers

- `Video Iterative Sampler`
- `Image Iterative Sampler`
- `Video Interpolate Sampler`
- `Explorer Conditioning Sampler`

### Integrations

- `Wave Style Model Apply`
- `Wave CLIP Vision Encode`
- `Wave IPAdapter Controller`
- `Wave IPAdapter Advanced`
- `Wave ControlNet Apply`
- `Wave ControlNet Controller`

### Utility

- `Wave Visualizer`
- `Iterative Checkpoint Controller`
- `Checkpoint Preview Loader`

## Typical Workflow Patterns

### 1) Video Iteration

`Load video latents -> Video Wave Controller -> (optional Wave integrations) -> Video Iterative Sampler -> VAE Decode -> Save/Preview`

### 2) Video Morph (A to B)

`Video A + Video B -> Interpolation Wave Controller -> Video Interpolate Sampler -> VAE Decode -> Save/Preview`

### 3) Conditioning Explorer

`Positive A (+ B, optional C) -> Explorer Conditioning Wave Controller -> Explorer Conditioning Sampler -> VAE Decode -> Grid/Video`

## Checkpointing (Short Version)

Use `Iterative Checkpoint Controller` when rendering long sequences.

- Saves progress during the run
- Lets you resume after interruption
- Can restore workflow state from preview metadata image

## Current Example Assets

- Demo GIFs: `examples/assets/demo.gif`, `examples/assets/demo2.gif`
- Input clips: `examples/assets/Example_Input_01_Man_Walk.mp4`, `examples/assets/Example_Input_02_Snake.mp4`
- Interpolation clips: `examples/assets/Example_input_walk_A.mp4`, `examples/assets/Example_input_walk_B.mp4`
- Reference images: `examples/assets/Reference_Image_01_crows.png`, `examples/assets/Reference_Image_02_flowers.png.png`

## Suggested Images/Videos To Add

These will make the repo easier to understand in 30 seconds:

1. Before/After loop for one video workflow
- 6-10 second side-by-side MP4 (`input` vs `output`)

2. Explorer A->B->C strip
- A single image grid showing key frames (start, 25%, 50%, 75%, end)

3. Wave Visualizer + output pair
- One screenshot of the visualizer and one matching output clip/frame set

4. Checkpoint resume proof
- Very short clip showing interrupted run and resumed completion

5. Interpolation showcase
- A->B source clips + final morph result (same prompt/seed details in caption)

6. “Starter recipes” thumbnails
- 3 small images linked to workflow files:
  - Stable video stylization
  - Heavy trippy diffusion
  - Smooth interpolation

## Notes

- This project is still evolving quickly.
- If something breaks after a ComfyUI update, open an issue and include your workflow + error log.

## Support

- Issues: https://github.com/Kentskooking/kentskooking-nodes/issues
- Discussions: https://github.com/Kentskooking/kentskooking-nodes/discussions

## Attribution

This pack is inspired by and references ideas/code from:

- Comfyui-FeedbackSampler by pizurny
- comfyui-mixlab-nodes by MixLabPro
- ComfyUI-KJNodes by kijai
- ComfyUI_IPAdapter_plus by cubiq
- ComfyUI Inspire Pack by ltdrdata
