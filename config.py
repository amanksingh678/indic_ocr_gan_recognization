"""
Central configuration for the Synthetic Scene Text Generation Pipeline.
"""
from dataclasses import dataclass, field
from typing import Tuple, List
import os


@dataclass
class PipelineConfig:
    """All tuneable parameters for the pipeline."""

    # --- Paths ---
    corpus_path: str = "hin-in_web_2015_1M-sentences.txt"
    fonts_dir: str = "fonts"
    output_dir: str = "output"

    # --- Generation ---
    num_images: int = 1000
    image_width_range: Tuple[int, int] = (320, 800)
    image_height_range: Tuple[int, int] = (80, 300)
    generation_mode: str = "word"  # "word" or "line"

    # --- Font ---
    font_size_range: Tuple[int, int] = (24, 72)

    # --- Text Rendering ---
    text_color_auto_contrast: bool = True
    text_shadow_prob: float = 0.3
    text_outline_prob: float = 0.3
    rotation_range: Tuple[float, float] = (-12.0, 12.0)
    max_words_per_image: int = 5  # used in "word" mode for multi-word scenes

    # --- Augmentation probabilities ---
    aug_gaussian_blur_prob: float = 0.3
    aug_motion_blur_prob: float = 0.15
    aug_noise_prob: float = 0.3
    aug_brightness_prob: float = 0.3
    aug_perspective_prob: float = 0.2
    aug_jpeg_prob: float = 0.25
    aug_elastic_prob: float = 0.15

    # --- Augmentation parameters ---
    blur_kernel_range: Tuple[int, int] = (3, 7)
    noise_intensity_range: Tuple[float, float] = (5.0, 25.0)
    brightness_alpha_range: Tuple[float, float] = (0.7, 1.3)
    brightness_beta_range: Tuple[int, int] = (-30, 30)
    jpeg_quality_range: Tuple[int, int] = (30, 85)
    perspective_magnitude: float = 0.05

    # --- Output ---
    image_format: str = "jpg"  # "jpg" or "png"
    save_sidecar_txt: bool = True  # per-image .txt label file

    # --- System ---
    seed: int = 42
    workers: int = 1

    @property
    def images_dir(self) -> str:
        return os.path.join(self.output_dir, "images")

    @property
    def annotations_path(self) -> str:
        return os.path.join(self.output_dir, "annotations.json")

    @property
    def labels_dir(self) -> str:
        return os.path.join(self.output_dir, "labels")
