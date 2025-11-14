import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

class WaveVisualizer:
    """
    Visualizes triangle wave parameters as a color-coded graph.
    Supports both basic and advanced triangle wave controllers.

    Basic: Shows steps, denoise, zoom, and CLIP strength oscillations.
    Advanced: Shows start_step, end_step, derived steps, zoom, and CLIP strength.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preview_frames": ("INT", {"default": 100, "min": 10, "max": 10000, "step": 1}),
                "graph_width": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64}),
                "graph_height": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 64}),
                "wave_config": ("TRIANGLE_WAVE_CONFIG",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("visualization",)
    FUNCTION = "visualize"
    CATEGORY = "kentskooking/visualization"

    def triangle_wave(self, position, cycle_length, min_val, max_val):
        """Calculate triangle wave value at given position."""
        half_cycle = cycle_length / 2.0
        if position <= half_cycle:
            t = position / half_cycle
        else:
            t = (cycle_length - position) / half_cycle
        return min_val + (max_val - min_val) * t

    def normalize_values(self, values, display_min=0.0, display_max=1.0):
        """Normalize values to 0-1 range for display."""
        arr = np.array(values)
        val_min = arr.min()
        val_max = arr.max()

        if val_max - val_min < 0.001:
            return np.ones_like(arr) * 0.5

        normalized = (arr - val_min) / (val_max - val_min)
        return normalized

    def visualize(self, preview_frames, graph_width, graph_height, wave_config):
        """
        Generate visualization of triangle wave parameters.
        Supports both basic and advanced triangle wave controllers.
        """
        cycle_length = wave_config["cycle_length"]
        is_advanced = wave_config.get("controller_variant") == "advanced"

        steps_values = []
        denoise_values = []
        zoom_values = []
        clip_values = []
        start_step_values = []
        end_step_values = []

        for frame_idx in range(preview_frames):
            position = frame_idx % cycle_length

            if is_advanced:
                # Advanced controller: calculate start/end steps
                step_floor = max(1, wave_config["step_floor"])
                start_at = wave_config["start_at_step"]
                max_end = max(start_at + step_floor, wave_config["end_at_step"])
                min_end = start_at + step_floor

                current_end = self.triangle_wave(position, cycle_length, min_end, max_end)
                end_at = int(round(max(current_end, min_end)))
                steps = max(step_floor, end_at - start_at)

                steps_values.append(steps)
                start_step_values.append(start_at)
                end_step_values.append(end_at)
            else:
                # Basic controller: use min/max steps and denoise
                steps = self.triangle_wave(position, cycle_length,
                                          wave_config["steps_min"], wave_config["steps_max"])
                denoise = self.triangle_wave(position, cycle_length,
                                            wave_config["denoise_min"], wave_config["denoise_max"])
                steps_values.append(steps)
                denoise_values.append(denoise)

            zoom = self.triangle_wave(position, cycle_length,
                                     wave_config["zoom_min"], wave_config["zoom_max"])
            clip_str = self.triangle_wave(position, cycle_length,
                                         wave_config["clip_strength_min"], wave_config["clip_strength_max"])

            zoom_values.append(zoom)
            clip_values.append(clip_str)

        img = Image.new('RGB', (graph_width, graph_height), color=(20, 20, 20))
        draw = ImageDraw.Draw(img)

        padding = 60
        graph_area_width = graph_width - (padding * 2)
        graph_area_height = graph_height - (padding * 2)

        draw.rectangle([padding, padding, graph_width - padding, graph_height - padding],
                      outline=(60, 60, 60), width=2)

        num_cycles = int(np.ceil(preview_frames / cycle_length))
        for i in range(1, num_cycles):
            cycle_x = padding + int((i * cycle_length / preview_frames) * graph_area_width)
            draw.line([(cycle_x, padding), (cycle_x, graph_height - padding)],
                     fill=(40, 40, 40), width=1)

        colors = {
            'steps': (255, 255, 0),
            'denoise': (255, 50, 50),
            'zoom': (50, 150, 255),
            'clip': (200, 100, 255),
            'start_step': (100, 255, 100),
            'end_step': (255, 150, 100),
        }

        if is_advanced:
            # Advanced controller: show start_step, end_step, steps (derived), zoom, clip
            step_min = min(steps_values)
            step_max = max(steps_values)
            parameters = [
                ('Start Step', start_step_values, colors['start_step'],
                 wave_config['start_at_step'], wave_config['start_at_step']),
                ('End Step', end_step_values, colors['end_step'],
                 wave_config['start_at_step'] + wave_config['step_floor'], wave_config['end_at_step']),
                ('Steps (derived)', steps_values, colors['steps'], step_min, step_max),
                ('Zoom', zoom_values, colors['zoom'], wave_config['zoom_min'], wave_config['zoom_max']),
                ('CLIP Strength', clip_values, colors['clip'], wave_config['clip_strength_min'], wave_config['clip_strength_max']),
            ]
        else:
            # Basic controller: show steps, denoise, zoom, clip
            parameters = [
                ('Steps', steps_values, colors['steps'], wave_config['steps_min'], wave_config['steps_max']),
                ('Denoise', denoise_values, colors['denoise'], wave_config['denoise_min'], wave_config['denoise_max']),
                ('Zoom', zoom_values, colors['zoom'], wave_config['zoom_min'], wave_config['zoom_max']),
                ('CLIP Strength', clip_values, colors['clip'], wave_config['clip_strength_min'], wave_config['clip_strength_max']),
            ]

        band_height = graph_area_height / len(parameters)
        band_padding = min(30, band_height * 0.2)
        line_width = 3

        for i in range(1, len(parameters)):
            y = padding + i * band_height
            draw.line([(padding, y), (graph_width - padding, y)], fill=(40, 40, 40), width=1)

        def draw_wave(values, color, label, min_val, max_val, band_idx):
            normalized = self.normalize_values(values)
            band_top = padding + band_idx * band_height + band_padding
            usable_height = max(band_height - (band_padding * 2), 1)
            points = []
            for i, val in enumerate(normalized):
                x = padding + int((i / (preview_frames - 1)) * graph_area_width)
                y = band_top + int((1 - val) * usable_height)
                points.append((x, y))

            if len(points) > 1:
                draw.line(points, fill=color, width=line_width)
            label_text = f"{label}: {min_val:.2f}-{max_val:.2f}" if isinstance(min_val, float) or isinstance(max_val, float) else f"{label}: {min_val}-{max_val}"
            text_y = band_top - band_padding * 0.6
            draw.text((padding + 10, text_y), label_text, fill=color)

        for idx, (label, values, color, min_val, max_val) in enumerate(parameters):
            draw_wave(values, color, label, min_val, max_val, idx)

        legend_y = 20
        legend_x = padding
        for i, (label, _, color, _, _) in enumerate(parameters):
            x_offset = i * 160
            draw.rectangle([legend_x + x_offset, legend_y,
                          legend_x + x_offset + 15, legend_y + 15],
                         fill=color)
            draw.text((legend_x + x_offset + 20, legend_y),
                      label, fill=(200, 200, 200))

        title = f"Triangle Wave Preview - Cycle: {cycle_length} frames - Preview: {preview_frames} frames"
        title_bbox = draw.textbbox((0, 0), title)
        title_width = title_bbox[2] - title_bbox[0]
        draw.text(((graph_width - title_width) // 2, graph_height - 30), title, fill=(150, 150, 150))

        img_np = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).unsqueeze(0)

        return (img_tensor,)


NODE_CLASS_MAPPINGS = {
    "WaveVisualizer": WaveVisualizer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WaveVisualizer": "Wave Visualizer"
}
