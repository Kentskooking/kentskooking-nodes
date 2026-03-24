# kentskooking-nodes

A nodepack that applies deforum style animation effects to images and videos.

## Overview

This pack adds wave-driven iterative tools for ComfyUI. It is built around frame-by-frame and iteration-by-iteration workflows that let you push motion, morphing, feedback, and stylization over time instead of treating each frame as an isolated render.

It currently covers four main workflow styles:

- Video iteration and stylization
- Image iteration
- Video interpolation and morphing
- Conditioning exploration across prompt states

## Included Nodes

### Controllers

- `Video Wave Controller`
- `Image Wave Controller`
- `Interpolation Wave Controller`
- `Explorer Conditioning Wave Controller`
- `Wave CLIP Vision Encode`
- `Wave Style Model Apply`
- `Wave IPAdapter Controller`
- `Wave ControlNet Apply`
- `Wave ControlNet Controller`

### Samplers

- `Video Iterative Sampler`
- `Image Iterative Sampler`
- `Video Interpolate Sampler`
- `Explorer Conditioning Sampler`

### Checkpointing and Utility

- `Iterative Checkpoint Controller`
- `Checkpoint Preview Loader`
- `Wave Visualizer`
- `Wave IPAdapter Advanced`

## Example Workflows

Included workflows live in `examples/workflows/`:

- `FLUX Video Iterative WF_kentskooking-nodes.json`
- `Image Iterative Sampler - zImage.json`
- `Video Interpolate WF - FLUX.json`
- `Video Interpolate WF (SDXL).json`
- `Video Interpolate WF (SD 1.5).json`
- `kentskooking_nodes_all.json`
- `WF_using_standard_nodes.json`

Included assets live in `examples/assets/` and include demo GIFs, reference images, and sample input clips.

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Kentskooking/kentskooking-nodes.git
```

Restart ComfyUI after cloning.

## Dependencies

Core Python dependencies are listed in `pyproject.toml` and are standard ComfyUI-side packages:

- `torch`
- `numpy>=1.25.0`
- `Pillow`
- `safetensors>=0.4.2`

Optional dependency:

- `ComfyUI_IPAdapter_plus` is only needed for the IPAdapter-related nodes.

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus.git comfyui_ipadapter_plus
```

## Typical Workflow Patterns

### Video Iteration

`Load video latents -> Video Wave Controller -> optional wave integrations -> Video Iterative Sampler -> VAE Decode -> Save or Preview`

### Image Iteration

`Load image latent -> Image Wave Controller -> optional wave integrations -> Image Iterative Sampler -> VAE Decode -> Save or Preview`

### Video Morphing

`Video A + Video B -> Interpolation Wave Controller -> Video Interpolate Sampler -> VAE Decode -> Save or Preview`

### Conditioning Exploration

`Positive A (+ optional B/C) -> Explorer Conditioning Wave Controller -> Explorer Conditioning Sampler -> VAE Decode -> Grid or Video`

## Checkpointing

Use `Iterative Checkpoint Controller` for long renders.

- Saves rolling progress during a run
- Lets you resume after interruption
- Can restore workflow state from preview metadata with `Checkpoint Preview Loader`

## Notes

- The project is still evolving and the workflows are the fastest way to get oriented.
- FLUX, SDXL, and SD 1.5 example workflows are all included.
- A small frontend script is bundled under `js/` for node width behavior in ComfyUI.

## Support

- Issues: https://github.com/Kentskooking/kentskooking-nodes/issues
- Discussions: https://github.com/Kentskooking/kentskooking-nodes/discussions

## License

This repository is licensed under GPL-3.0. See `LICENSE` and `NOTICE` for project and third-party attribution details.
