import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ..utils.kentskooking_utils import calculate_wave, normalize_values

class WaveVisualizer:
    """
    Visualizes wave parameters as a color-coded graph.
    Supports Video, Image, and Interpolation wave configurations.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "wave_config": ("WAVE_CONFIG",),
                "width": ("INT", {"default": 800, "min": 100, "max": 2000}),
                "height": ("INT", {"default": 400, "min": 100, "max": 2000}),
                "wave_selector": (["all", "steps", "denoise", "zoom", "clip_strength", "controlnet_strength", "ipadapter_weight"], {"default": "all"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("visualization",)
    FUNCTION = "visualize"
    CATEGORY = "kentskooking/visualization"

    def visualize(self, wave_config, width, height, wave_selector):
        # 1. Determine Context (Cycle Length, Mode)
        cycle_length = wave_config.get("cycle_length", 60)
        wave_type = wave_config.get("wave_type", "triangle")
        controller_type = wave_config.get("controller_type", "video") # video, image, interpolation
        
        # Determine preview duration
        if controller_type == "interpolation":
            # For interpolation, the cycle IS the overlap duration. Show 1 full pass.
            preview_frames = cycle_length
            title_prefix = "Interpolation Morph"
        elif controller_type == "image":
            # For image, show a few cycles to illustrate progression
            preview_frames = cycle_length * wave_config.get("cycle_count", 3)
            title_prefix = "Image Iterative"
        else:
            # For video, show 2 cycles by default
            preview_frames = cycle_length * 2
            title_prefix = "Video Wave"

        # 2. Collect Data Series
        tracks = []

        # --- STEPS (Video/Image) ---
        if "step_floor" in wave_config:
            steps_vals = []
            start_vals = []
            end_vals = []
            
            step_floor = max(1, wave_config.get("step_floor", 1))
            start_at = wave_config.get("start_at_step", 0)
            end_target = wave_config.get("end_at_step", 20)
            min_end = start_at + step_floor
            max_end = max(min_end, end_target)

            for i in range(preview_frames):
                pos = i % cycle_length
                # Image controller linear step ramp logic vs Video wave logic
                if controller_type == "image":
                    # Image: Steps increase linearly 0 -> length within cycle
                    current_steps = step_floor + pos
                    current_end = start_at + current_steps
                    actual_end = min(current_end, end_target)
                    steps_vals.append(actual_end - start_at)
                    start_vals.append(start_at)
                    end_vals.append(actual_end)
                else:
                    # Video Wave logic
                    current_end = calculate_wave(wave_type, pos, cycle_length, min_end, max_end)
                    end_at = int(round(max(current_end, min_end)))
                    steps_vals.append(max(step_floor, end_at - start_at))
                    start_vals.append(start_at)
                    end_vals.append(end_at)

            tracks.append({"id": "steps", "label": "Steps", "values": steps_vals, "color": (255, 255, 0), "range": (min(steps_vals), max(steps_vals))})
            if wave_selector == "steps": # Show details only if focused
                tracks.append({"id": "steps_start", "label": "Start Step", "values": start_vals, "color": (100, 255, 100), "range": (start_at, start_at)})
                tracks.append({"id": "steps_end", "label": "End Step", "values": end_vals, "color": (255, 150, 100), "range": (min(end_vals), max(end_vals))})

        # --- DENOISE (Interpolation/Basic) ---
        if "denoise_min" in wave_config:
            denoise_vals = []
            d_min = wave_config.get("denoise_min", 0.0)
            d_max = wave_config.get("denoise_max", 1.0)
            
            for i in range(preview_frames):
                pos = i % cycle_length
                # Special handling for Interpolation Seam Smoothing
                val = calculate_wave(wave_type, pos, cycle_length, d_min, d_max)
                if controller_type == "interpolation":
                    ramp_len = max(1, int(cycle_length * 0.1))
                    if pos < ramp_len:
                        val *= (pos / ramp_len)
                    elif pos >= cycle_length - ramp_len:
                        val *= ((cycle_length - 1 - pos) / ramp_len)
                
                denoise_vals.append(val)
            
            tracks.append({"id": "denoise", "label": "Denoise", "values": denoise_vals, "color": (255, 50, 50), "range": (min(denoise_vals), max(denoise_vals))})

        # --- ZOOM (All) ---
        if "zoom_min" in wave_config or "zoom_rate" in wave_config:
            zoom_vals = []
            z_min = wave_config.get("zoom_min", 1.0)
            z_max = wave_config.get("zoom_max", 1.0)
            z_rate = wave_config.get("zoom_rate", 1.0) # For image exponential
            
            for i in range(preview_frames):
                pos = i % cycle_length
                if controller_type == "image":
                    # Exponential cumulative zoom
                    # This visualizes the zoom factor APPLIED at that step
                    val = z_rate ** (pos + 1)
                else:
                    val = calculate_wave(wave_type, pos, cycle_length, z_min, z_max)
                zoom_vals.append(val)
            
            tracks.append({"id": "zoom", "label": "Zoom", "values": zoom_vals, "color": (50, 150, 255), "range": (min(zoom_vals), max(zoom_vals))})

        # --- CLIP STRENGTH ---
        if "clip_strength_min" in wave_config:
            clip_vals = []
            c_min = wave_config.get("clip_strength_min", 0.0)
            c_max = wave_config.get("clip_strength_max", 1.0)
            for i in range(preview_frames):
                pos = i % cycle_length
                if controller_type == "image":
                    # Linear ramp per cycle
                    t = pos / max(1, cycle_length - 1)
                    val = c_min + (c_max - c_min) * t
                else:
                    val = calculate_wave(wave_type, pos, cycle_length, c_min, c_max)
                clip_vals.append(val)
            tracks.append({"id": "clip_strength", "label": "CLIP Strength", "values": clip_vals, "color": (200, 100, 255), "range": (c_min, c_max)})

        # --- CONTROLNET ---
        if "controlnet_strength_min" in wave_config:
            cn_vals_A = []
            cn_vals_B = [] # For interpolation
            cn_min = wave_config.get("controlnet_strength_min", 0.0)
            cn_max = wave_config.get("controlnet_strength_max", 1.0)
            
            for i in range(preview_frames):
                pos = i % cycle_length
                
                if controller_type == "interpolation":
                    # Interpolation Logic (A fades out, B fades in)
                    t = pos / max(1, cycle_length - 1)
                    val_A = cn_max * (1.0 - t) + cn_min * t
                    val_B = cn_min * (1.0 - t) + cn_max * t
                    cn_vals_A.append(val_A)
                    cn_vals_B.append(val_B)
                else:
                    val = calculate_wave(wave_type, pos, cycle_length, cn_min, cn_max)
                    cn_vals_A.append(val)

            if controller_type == "interpolation":
                tracks.append({"id": "controlnet_strength", "label": "CN Strength A (Out)", "values": cn_vals_A, "color": (255, 100, 0), "range": (cn_min, cn_max)})
                tracks.append({"id": "controlnet_strength", "label": "CN Strength B (In)", "values": cn_vals_B, "color": (255, 200, 0), "range": (cn_min, cn_max)})
            else:
                tracks.append({"id": "controlnet_strength", "label": "CN Strength", "values": cn_vals_A, "color": (255, 150, 0), "range": (cn_min, cn_max)})

        # --- IP ADAPTER ---
        if "ipadapter_weight_min" in wave_config:
            ip_vals = []
            ip_min = wave_config.get("ipadapter_weight_min", 0.0)
            ip_max = wave_config.get("ipadapter_weight_max", 1.0)
            for i in range(preview_frames):
                pos = i % cycle_length
                val = calculate_wave(wave_type, pos, cycle_length, ip_min, ip_max)
                ip_vals.append(val)
            tracks.append({"id": "ipadapter_weight", "label": "IPAdapter Weight", "values": ip_vals, "color": (0, 255, 200), "range": (ip_min, ip_max)})


        # 3. Filter Tracks based on Selector
        if wave_selector != "all":
            filtered_tracks = [t for t in tracks if t["id"] == wave_selector or t["id"] == f"{wave_selector}_start" or t["id"] == f"{wave_selector}_end"]
            if filtered_tracks:
                tracks = filtered_tracks
            else:
                # If selector matches nothing (e.g. no controlnet config but 'controlnet' selected), show empty
                tracks = []

        # 4. Draw Graph
        if not tracks:
            # Draw empty placeholder
            img = Image.new('RGB', (width, height), color=(20, 20, 20))
            draw = ImageDraw.Draw(img)
            draw.text((width//2 - 50, height//2), "No Data", fill=(100, 100, 100))
            img_np = np.array(img).astype(np.float32) / 255.0
            return (torch.from_numpy(img_np).unsqueeze(0),)

        img = Image.new('RGB', (width, height), color=(20, 20, 20))
        draw = ImageDraw.Draw(img)
        
        padding = 60
        graph_w = width - (padding * 2)
        graph_h = height - (padding * 2)
        
        # Draw Axis Box
        draw.rectangle([padding, padding, width - padding, height - padding], outline=(60, 60, 60), width=2)
        
        # Draw Cycle Lines
        num_cycles = int(np.ceil(preview_frames / cycle_length))
        for i in range(1, num_cycles):
            cx = padding + int((i * cycle_length / preview_frames) * graph_w)
            if cx < width - padding:
                draw.line([(cx, padding), (cx, height - padding)], fill=(40, 40, 40), width=1)

        # Calculate Band Layout
        num_bands = len(tracks)
        band_h = graph_h / num_bands
        
        for idx, track in enumerate(tracks):
            band_top = padding + idx * band_h
            band_bottom = band_top + band_h
            
            # Draw Band Separator
            if idx > 0:
                draw.line([(padding, band_top), (width - padding, band_top)], fill=(40, 40, 40), width=1)
            
            # Normalize and Plot
            values = track["values"]
            v_min, v_max = track["range"]
            
            # Safety for flat lines
            denom = v_max - v_min
            if abs(denom) < 0.0001: denom = 1.0
            
            points = []
            for i, val in enumerate(values):
                x = padding + int((i / (preview_frames - 1)) * graph_w)
                # Y is inverted (0 at top)
                # Normalize 0..1 within band
                norm_val = (val - v_min) / denom
                y = band_bottom - 10 - int(norm_val * (band_h - 20))
                points.append((x, y))
            
            if len(points) > 1:
                draw.line(points, fill=track["color"], width=3)
            
            # Labels
            label_text = f"{track['label']}: {v_min:.2f} - {v_max:.2f}"
            draw.text((padding + 10, band_top + 5), label_text, fill=track["color"])

        # Title
        title = f"{title_prefix} - {wave_type.title()} Wave - Cycle: {cycle_length} frames"
        draw.text((padding, 10), title, fill=(150, 150, 150))

        img_np = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).unsqueeze(0)

        return (img_tensor,)


NODE_CLASS_MAPPINGS = {
    "WaveVisualizer": WaveVisualizer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WaveVisualizer": "Wave Visualizer"
}