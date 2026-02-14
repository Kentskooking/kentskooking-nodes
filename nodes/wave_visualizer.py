import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ..utils.kentskooking_utils import calculate_wave, normalize_values, calculate_explorer_path_strengths

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
                "wave_selector": ([
                    "all",
                    "steps",
                    "denoise",
                    "zoom",
                    "clip_strength",
                    "controlnet_strength",
                    "ipadapter_weight",
                    "positive_a_strength",
                    "positive_b_strength",
                    "positive_c_strength"
                ], {"default": "all"}),
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
        elif controller_type == "explorer_conditioning":
            # Explorer traversal is a single pass across strength ranges.
            c_start_preview = float(wave_config.get("positive_c_start", 0.0))
            c_end_preview = float(wave_config.get("positive_c_end", 0.0))
            has_positive_c = abs(c_start_preview) > 1e-6 or abs(c_end_preview) > 1e-6
            loop_requested = bool(wave_config.get("loop_video", False))
            use_loop = loop_requested and has_positive_c
            preview_frames = 240 if use_loop else (180 if has_positive_c else 120)
            cycle_length = preview_frames
            title_prefix = "Explorer Conditioning"
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

        # --- EXPLORER CONDITIONING STRENGTHS ---
        if controller_type == "explorer_conditioning" or "positive_a_start" in wave_config or "positive_b_start" in wave_config:
            curve_type = wave_config.get("curve_type", "linear")
            a_start = float(wave_config.get("positive_a_start", 0.0))
            a_end = float(wave_config.get("positive_a_end", 1.0))
            b_start = float(wave_config.get("positive_b_start", 1.0))
            b_end = float(wave_config.get("positive_b_end", 0.0))
            c_start = float(wave_config.get("positive_c_start", 0.0))
            c_end = float(wave_config.get("positive_c_end", 0.0))
            has_positive_c = abs(c_start) > 1e-6 or abs(c_end) > 1e-6
            loop_requested = bool(wave_config.get("loop_video", False))
            use_loop = loop_requested and has_positive_c

            a_vals = []
            b_vals = []
            c_vals = []
            for i in range(preview_frames):
                a_strength, b_strength, c_strength = calculate_explorer_path_strengths(
                    iteration_idx=i,
                    iteration_count=preview_frames,
                    curve_type=curve_type,
                    positive_a_start=a_start,
                    positive_a_end=a_end,
                    positive_b_start=b_start,
                    positive_b_end=b_end,
                    positive_c_start=c_start,
                    positive_c_end=c_end,
                    has_positive_c=has_positive_c,
                    loop_video=use_loop,
                )
                a_vals.append(a_strength)
                b_vals.append(b_strength)
                if has_positive_c:
                    c_vals.append(c_strength)

            tracks.append({
                "id": "positive_a_strength",
                "label": "Positive A Strength",
                "values": a_vals,
                "color": (255, 120, 0),
                "range": (min(a_vals), max(a_vals)),
                "overlay_group": "explorer_strengths",
            })
            tracks.append({
                "id": "positive_b_strength",
                "label": "Positive B Strength",
                "values": b_vals,
                "color": (0, 200, 255),
                "range": (min(b_vals), max(b_vals)),
                "overlay_group": "explorer_strengths",
            })
            if has_positive_c:
                tracks.append({
                    "id": "positive_c_strength",
                    "label": "Positive C Strength",
                    "values": c_vals,
                    "color": (120, 255, 120),
                    "range": (min(c_vals), max(c_vals)),
                    "overlay_group": "explorer_strengths",
                })


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
        
        padding_x = 60
        padding_top = 60
        padding_bottom = 90
        graph_w = width - (padding_x * 2)
        graph_h = height - (padding_top + padding_bottom)
        
        # Draw Axis Box
        draw.rectangle(
            [padding_x, padding_top, width - padding_x, height - padding_bottom],
            outline=(60, 60, 60),
            width=2,
        )
        
        # Draw Cycle Lines
        num_cycles = int(np.ceil(preview_frames / cycle_length))
        for i in range(1, num_cycles):
            cx = padding_x + int((i * cycle_length / preview_frames) * graph_w)
            if cx < width - padding_x:
                draw.line([(cx, padding_top), (cx, height - padding_bottom)], fill=(40, 40, 40), width=1)

        # Calculate Band Layout (supports overlay groups sharing one axis/band)
        band_groups = []
        overlay_group_index = {}

        for track in tracks:
            overlay_group = track.get("overlay_group")
            if overlay_group:
                if overlay_group not in overlay_group_index:
                    overlay_group_index[overlay_group] = len(band_groups)
                    band_groups.append({"tracks": [track], "overlay_group": overlay_group})
                else:
                    band_groups[overlay_group_index[overlay_group]]["tracks"].append(track)
            else:
                band_groups.append({"tracks": [track], "overlay_group": None})

        num_bands = len(band_groups)
        band_h = graph_h / num_bands
        legend_entries = []
        
        for idx, band in enumerate(band_groups):
            band_top = padding_top + idx * band_h
            band_bottom = band_top + band_h
            
            # Draw Band Separator
            if idx > 0:
                draw.line([(padding_x, band_top), (width - padding_x, band_top)], fill=(40, 40, 40), width=1)

            band_tracks = band["tracks"]
            if band["overlay_group"] is not None:
                v_min = min(min(t["values"]) for t in band_tracks)
                v_max = max(max(t["values"]) for t in band_tracks)
            else:
                v_min, v_max = band_tracks[0]["range"]

            # Safety for flat lines
            denom = v_max - v_min
            if abs(denom) < 0.0001:
                denom = 1.0

            for track_idx, track in enumerate(band_tracks):
                values = track["values"]
                points = []
                for i, val in enumerate(values):
                    x = padding_x + int((i / (preview_frames - 1)) * graph_w)
                    # Y is inverted (0 at top)
                    # Normalize 0..1 within band
                    norm_val = (val - v_min) / denom
                    y = band_bottom - 10 - int(norm_val * (band_h - 20))
                    points.append((x, y))

                if len(points) > 1:
                    draw.line(points, fill=track["color"], width=3)

                # Labels
                if band["overlay_group"] is not None:
                    label_text = f"{track['label']}: {min(values):.2f} - {max(values):.2f}"
                else:
                    label_text = f"{track['label']}: {v_min:.2f} - {v_max:.2f}"

                legend_entries.append((track["color"], label_text))

        # Draw separate legend below plot area (prevents text/curve overlap)
        legend_y = height - padding_bottom + 12
        legend_x = padding_x
        legend_line_h = 16
        for color, text in legend_entries:
            try:
                bbox = draw.textbbox((0, 0), text)
                text_w = bbox[2] - bbox[0]
            except Exception:
                text_w = len(text) * 7

            item_w = 16 + text_w + 20
            if legend_x + item_w > width - padding_x:
                legend_x = padding_x
                legend_y += legend_line_h

            draw.line([(legend_x, legend_y + 6), (legend_x + 12, legend_y + 6)], fill=color, width=3)
            draw.text((legend_x + 16, legend_y), text, fill=color)
            legend_x += item_w

        # Title
        if controller_type == "explorer_conditioning":
            curve_label = wave_config.get("curve_type", "linear").replace("_", " ").title()
            c_start_title = float(wave_config.get("positive_c_start", 0.0))
            c_end_title = float(wave_config.get("positive_c_end", 0.0))
            has_positive_c = abs(c_start_title) > 1e-6 or abs(c_end_title) > 1e-6
            loop_requested = bool(wave_config.get("loop_video", False))
            use_loop = loop_requested and has_positive_c
            if use_loop:
                path_label = "A->B->C->A"
            elif has_positive_c:
                path_label = "A->B->C"
            else:
                path_label = "A->B"
            title = f"{title_prefix} - {curve_label} Curve - Path: {path_label} - Traversal: {preview_frames} frames"
        else:
            title = f"{title_prefix} - {wave_type.title()} Wave - Cycle: {cycle_length} frames"
        draw.text((padding_x, 10), title, fill=(150, 150, 150))

        img_np = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).unsqueeze(0)

        return (img_tensor,)


NODE_CLASS_MAPPINGS = {
    "WaveVisualizer": WaveVisualizer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WaveVisualizer": "Wave Visualizer"
}
