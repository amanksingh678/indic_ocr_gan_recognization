"""
Augmentor — applies realistic post-rendering augmentations using OpenCV.
Transforms both images and bounding box coordinates.
"""
import numpy as np
import cv2
from typing import List, Dict, Tuple
from PIL import Image


class Augmentor:
    """Applies configurable augmentations to rendered scene text images."""

    def __init__(self, config):
        """config: PipelineConfig instance."""
        self.cfg = config
        self.rng = np.random.RandomState(config.seed)

    def augment(self, image: Image.Image,
                annotations: List[Dict]) -> Tuple[Image.Image, List[Dict]]:
        """
        Apply random augmentations to image and update bbox annotations.
        Returns (augmented_image, updated_annotations).
        """
        img = np.array(image)
        anns = [a.copy() for a in annotations]
        for a in anns:
            a["bbox"] = list(a["bbox"])  # ensure mutable

        # Gaussian blur
        if self.rng.random() < self.cfg.aug_gaussian_blur_prob:
            ksize = self.rng.choice(range(self.cfg.blur_kernel_range[0],
                                          self.cfg.blur_kernel_range[1] + 1, 2))
            ksize = max(3, ksize if ksize % 2 == 1 else ksize + 1)
            img = cv2.GaussianBlur(img, (ksize, ksize), 0)

        # Motion blur
        if self.rng.random() < self.cfg.aug_motion_blur_prob:
            ksize = self.rng.randint(3, 8)
            kernel = np.zeros((ksize, ksize))
            if self.rng.random() < 0.5:
                kernel[ksize // 2, :] = 1.0  # horizontal
            else:
                kernel[:, ksize // 2] = 1.0  # vertical
            kernel /= ksize
            img = cv2.filter2D(img, -1, kernel)

        # Gaussian noise
        if self.rng.random() < self.cfg.aug_noise_prob:
            intensity = self.rng.uniform(*self.cfg.noise_intensity_range)
            noise = self.rng.normal(0, intensity, img.shape)
            img = np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)

        # Brightness / contrast jitter
        if self.rng.random() < self.cfg.aug_brightness_prob:
            alpha = self.rng.uniform(*self.cfg.brightness_alpha_range)
            beta = self.rng.randint(*self.cfg.brightness_beta_range)
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        # Perspective warp
        if self.rng.random() < self.cfg.aug_perspective_prob:
            img, anns = self._perspective_warp(img, anns)

        # JPEG compression artifacts
        if self.rng.random() < self.cfg.aug_jpeg_prob:
            quality = self.rng.randint(*self.cfg.jpeg_quality_range)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, encoded = cv2.imencode(".jpg", img, encode_param)
            img = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

        # Elastic distortion
        if self.rng.random() < self.cfg.aug_elastic_prob:
            img = self._elastic_distortion(img)

        # Final bbox clipping — ensure all bboxes are within image bounds
        h, w = img.shape[:2]
        clipped_anns = []
        for a in anns:
            x, y, bw, bh = a["bbox"]
            x = max(0, x)
            y = max(0, y)
            bw = min(bw, w - x)
            bh = min(bh, h - y)
            if bw > 0 and bh > 0:
                a["bbox"] = [x, y, bw, bh]
                clipped_anns.append(a)

        return Image.fromarray(img), clipped_anns

    def _perspective_warp(self, img: np.ndarray,
                          anns: List[Dict]) -> Tuple[np.ndarray, List[Dict]]:
        """Apply a slight random perspective warp."""
        h, w = img.shape[:2]
        mag = self.cfg.perspective_magnitude

        # Source corners
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        # Random perturbation
        dst = src + self.rng.uniform(-mag * min(w, h), mag * min(w, h),
                                      src.shape).astype(np.float32)
        # Ensure positive coordinates and correct dtype for OpenCV
        dst = np.clip(dst, 0, [w - 1, h - 1]).astype(np.float32)

        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(img, M, (w, h),
                                      borderMode=cv2.BORDER_REPLICATE)

        # Transform bboxes
        new_anns = []
        for ann in anns:
            bx, by, bw, bh = ann["bbox"]
            # 4 corners of bbox
            corners = np.float32([
                [bx, by], [bx + bw, by],
                [bx + bw, by + bh], [bx, by + bh]
            ]).reshape(-1, 1, 2)
            transformed = cv2.perspectiveTransform(corners, M).reshape(-1, 2)
            x_min = max(0, int(transformed[:, 0].min()))
            y_min = max(0, int(transformed[:, 1].min()))
            x_max = min(w, int(transformed[:, 0].max()))
            y_max = min(h, int(transformed[:, 1].max()))
            new_ann = ann.copy()
            new_ann["bbox"] = [x_min, y_min, x_max - x_min, y_max - y_min]
            new_anns.append(new_ann)

        return warped, new_anns

    def _elastic_distortion(self, img: np.ndarray,
                            alpha: float = 30.0,
                            sigma: float = 5.0) -> np.ndarray:
        """Apply elastic distortion using random displacement fields."""
        h, w = img.shape[:2]
        dx = cv2.GaussianBlur(
            (self.rng.rand(h, w).astype(np.float32) * 2 - 1),
            (0, 0), sigma) * alpha
        dy = cv2.GaussianBlur(
            (self.rng.rand(h, w).astype(np.float32) * 2 - 1),
            (0, 0), sigma) * alpha

        x, y = np.meshgrid(np.arange(w, dtype=np.float32),
                            np.arange(h, dtype=np.float32))
        map_x = x + dx
        map_y = y + dy

        distorted = cv2.remap(img, map_x, map_y,
                               interpolation=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)
        return distorted
