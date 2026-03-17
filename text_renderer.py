"""
Text renderer — renders Devanagari text onto background images with
font diversity, shadows, outlines, and slight transforms.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from typing import List, Dict, Tuple, Optional
from utils import contrasting_color, mean_color_of_region, luminance
import random as _random


class TextRenderer:
    """Renders text onto a background image and returns annotations."""

    def __init__(self, seed: int = 42,
                 shadow_prob: float = 0.3,
                 outline_prob: float = 0.3,
                 rotation_range: Tuple[float, float] = (-12.0, 12.0)):
        self.rng = _random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        self.shadow_prob = shadow_prob
        self.outline_prob = outline_prob
        self.rotation_range = rotation_range

    def render_line(self, background: Image.Image, text: str,
                    font: ImageFont.FreeTypeFont,
                    auto_contrast: bool = True) -> Tuple[Image.Image, List[Dict]]:
        """
        Render a full line of text on the background.
        Returns (composited_image, [annotation_dict]).
        """
        bg = background.copy().convert("RGBA")
        bg_w, bg_h = bg.size

        # Measure text
        dummy = Image.new("RGBA", (1, 1))
        dd = ImageDraw.Draw(dummy)
        bbox = dd.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        if tw <= 0 or th <= 0:
            return bg.convert("RGB"), []

        # Create text layer with padding for rotation
        pad = int(max(tw, th) * 0.3)
        txt_layer = Image.new("RGBA", (tw + 2 * pad, th + 2 * pad), (0, 0, 0, 0))
        draw = ImageDraw.Draw(txt_layer)

        # Determine text color
        bg_np = np.array(bg.convert("RGB"))
        # Sample region center
        sample_x = max(0, min(bg_w - tw, (bg_w - tw) // 2))
        sample_y = max(0, min(bg_h - th, (bg_h - th) // 2))
        if auto_contrast:
            region_color = mean_color_of_region(bg_np, sample_x, sample_y,
                                                 min(tw, bg_w), min(th, bg_h))
            text_color = contrasting_color(region_color, self.np_rng)
        else:
            text_color = (self.np_rng.randint(0, 256),
                          self.np_rng.randint(0, 256),
                          self.np_rng.randint(0, 256))

        text_pos = (pad - bbox[0], pad - bbox[1])

        # Optional shadow
        if self.rng.random() < self.shadow_prob:
            shadow_offset = self.rng.randint(1, 3)
            shadow_color = (0, 0, 0, 120)
            draw.text((text_pos[0] + shadow_offset, text_pos[1] + shadow_offset),
                      text, font=font, fill=shadow_color)

        # Optional outline
        if self.rng.random() < self.outline_prob:
            outline_color = (0, 0, 0, 200) if luminance(text_color) > 128 else (255, 255, 255, 200)
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    draw.text((text_pos[0] + dx, text_pos[1] + dy),
                              text, font=font, fill=outline_color)

        # Draw main text
        draw.text(text_pos, text, font=font, fill=(*text_color, 255))

        # Optional rotation
        angle = self.rng.uniform(*self.rotation_range)
        if abs(angle) > 0.5:
            txt_layer = txt_layer.rotate(angle, resample=Image.BICUBIC, expand=True)

        # Calculate paste position (centered, with random offset)
        tl_w, tl_h = txt_layer.size
        max_x = max(0, bg_w - tl_w)
        max_y = max(0, bg_h - tl_h)
        paste_x = self.rng.randint(0, max(1, max_x))
        paste_y = self.rng.randint(0, max(1, max_y))

        # Composite
        bg.paste(txt_layer, (paste_x, paste_y), txt_layer)
        result = bg.convert("RGB")

        # Compute tight bounding box from alpha channel
        txt_alpha = np.array(txt_layer.split()[-1])
        rows = np.any(txt_alpha > 0, axis=1)
        cols = np.any(txt_alpha > 0, axis=0)
        if rows.any() and cols.any():
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            ann_x = paste_x + int(cmin)
            ann_y = paste_y + int(rmin)
            ann_w = int(cmax - cmin + 1)
            ann_h = int(rmax - rmin + 1)
        else:
            ann_x, ann_y = paste_x, paste_y
            ann_w, ann_h = tw, th

        # Clip bbox to image bounds
        ann_x = max(0, ann_x)
        ann_y = max(0, ann_y)
        ann_w = min(ann_w, bg_w - ann_x)
        ann_h = min(ann_h, bg_h - ann_y)
        if ann_w <= 0 or ann_h <= 0:
            return result, []

        annotation = {
            "bbox": [ann_x, ann_y, ann_w, ann_h],
            "text": text,
            "language": "hi",
        }

        return result, [annotation]

    def render_words(self, background: Image.Image, words: List[str],
                     font_fn, auto_contrast: bool = True) -> Tuple[Image.Image, List[Dict]]:
        """
        Render multiple words at different positions on the background.
        font_fn: callable that returns a PIL ImageFont (called per word for diversity).
        Returns (composited_image, [annotation_dict, ...]).
        """
        result = background.copy().convert("RGB")
        annotations = []
        bg_w, bg_h = result.size

        # Divide the image into a grid to avoid overlaps
        n_words = len(words)
        cols = max(1, int(np.ceil(np.sqrt(n_words))))
        rows_grid = max(1, int(np.ceil(n_words / cols)))
        cell_w = bg_w // cols
        cell_h = bg_h // rows_grid

        for idx, word in enumerate(words):
            font = font_fn()
            row_i = idx // cols
            col_i = idx % cols

            # Create sub-region background
            x_start = col_i * cell_w
            y_start = row_i * cell_h
            sub_bg = result.crop((x_start, y_start,
                                  min(x_start + cell_w, bg_w),
                                  min(y_start + cell_h, bg_h)))

            if sub_bg.size[0] < 10 or sub_bg.size[1] < 10:
                continue

            sub_result, sub_anns = self.render_line(sub_bg, word, font, auto_contrast)

            # Paste back
            result.paste(sub_result, (x_start, y_start))

            # Adjust annotation coordinates
            for ann in sub_anns:
                ann["bbox"][0] += x_start
                ann["bbox"][1] += y_start
                annotations.append(ann)

        return result, annotations
