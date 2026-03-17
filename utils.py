"""
Utility helpers: random colors, bbox math, Perlin noise (pure numpy).
"""
import numpy as np
from typing import Tuple


# ---------------------------------------------------------------------------
# Pure-numpy Perlin noise (avoids C-extension dependency)
# ---------------------------------------------------------------------------

def _fade(t):
    return 6 * t**5 - 15 * t**4 + 10 * t**3


def _lerp(a, b, t):
    return a + t * (b - a)


def _gradient(h, x, y):
    vectors = np.array([[1, 1], [-1, 1], [1, -1], [-1, -1],
                         [1, 0], [-1, 0], [0, 1], [0, -1]])
    g = vectors[h % 8]
    return g[..., 0] * x + g[..., 1] * y


def perlin_noise(width: int, height: int, scale: float = 50.0,
                 seed: int | None = None) -> np.ndarray:
    """Generate a 2D Perlin noise array in [0, 1] of shape (height, width)."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()

    # permutation table
    p = np.arange(256, dtype=int)
    rng.shuffle(p)
    p = np.stack([p, p]).flatten()

    # coordinates
    lin_x = np.linspace(0, width / scale, width, endpoint=False)
    lin_y = np.linspace(0, height / scale, height, endpoint=False)
    x, y = np.meshgrid(lin_x, lin_y)

    xi = x.astype(int) % 256
    yi = y.astype(int) % 256
    xf = x - x.astype(int)
    yf = y - y.astype(int)

    u = _fade(xf)
    v = _fade(yf)

    n00 = _gradient(p[p[xi] + yi], xf, yf)
    n01 = _gradient(p[p[xi] + yi + 1], xf, yf - 1)
    n10 = _gradient(p[p[xi + 1] + yi], xf - 1, yf)
    n11 = _gradient(p[p[xi + 1] + yi + 1], xf - 1, yf - 1)

    x1 = _lerp(n00, n10, u)
    x2 = _lerp(n01, n11, u)
    noise = _lerp(x1, x2, v)

    # normalise to [0, 1]
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
    return noise


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

def random_color(rng: np.random.RandomState) -> Tuple[int, int, int]:
    """Return a random RGB color tuple."""
    return tuple(rng.randint(0, 256, size=3).tolist())


def luminance(color: Tuple[int, int, int]) -> float:
    """Perceived luminance (BT.601)."""
    return 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]


def contrasting_color(bg_color: Tuple[int, int, int],
                      rng: np.random.RandomState) -> Tuple[int, int, int]:
    """Pick a text color that contrasts with the background color."""
    bg_lum = luminance(bg_color)
    for _ in range(20):
        c = random_color(rng)
        if abs(luminance(c) - bg_lum) > 100:
            return c
    # fallback: black or white
    return (0, 0, 0) if bg_lum > 128 else (255, 255, 255)


def mean_color_of_region(img_array: np.ndarray,
                         x: int, y: int, w: int, h: int) -> Tuple[int, int, int]:
    """Mean RGB of a rectangular region."""
    region = img_array[y:y + h, x:x + w]
    if region.size == 0:
        return (128, 128, 128)
    mean = region.mean(axis=(0, 1))
    return tuple(int(v) for v in mean[:3])


# ---------------------------------------------------------------------------
# Bbox helpers
# ---------------------------------------------------------------------------

def clip_bbox(bbox, img_w, img_h):
    """Clip [x, y, w, h] to image bounds."""
    x, y, w, h = bbox
    x = max(0, x)
    y = max(0, y)
    w = min(w, img_w - x)
    h = min(h, img_h - y)
    return [x, y, w, h]
