"""
Procedural background generator — creates diverse synthetic backgrounds
simulating real-world scenes: rainy, sunny, brick, wood, water, foliage,
sunset, night, signboard, metallic, etc. No external datasets needed.
"""
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter
from typing import Tuple
from utils import perlin_noise, random_color


class BackgroundGenerator:
    """Generates highly diverse synthetic background images."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self._generators = [
            self._linear_gradient,
            self._radial_gradient,
            self._perlin_texture,
            self._geometric_pattern,
            self._paper_grain,
            self._cloudy_sky,
            self._stripe_pattern,
            self._rainy_scene,
            self._sunny_scene,
            self._brick_wall,
            self._wood_texture,
            self._water_surface,
            self._foliage_canopy,
            self._sunset_sky,
            self._night_scene,
            self._signboard,
            self._metallic_surface,
            self._concrete_wall,
            self._fabric_texture,
            self._bokeh_blur,
            self._composite,
        ]

    def generate(self, width: int, height: int) -> Image.Image:
        """Generate a random background of the given size."""
        gen_fn = self._generators[self.rng.randint(0, len(self._generators))]
        img_array = gen_fn(width, height)
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        return Image.fromarray(img_array, "RGB")

    # ------------------------------------------------------------------
    # BASIC BACKGROUNDS
    # ------------------------------------------------------------------

    def _linear_gradient(self, w: int, h: int) -> np.ndarray:
        """Linear gradient between two random colors."""
        c1 = np.array(random_color(self.rng), dtype=np.float64)
        c2 = np.array(random_color(self.rng), dtype=np.float64)
        direction = self.rng.choice(["horizontal", "vertical", "diagonal"])
        if direction == "horizontal":
            t = np.linspace(0, 1, w).reshape(1, w, 1)
            t = np.broadcast_to(t, (h, w, 1))
        elif direction == "vertical":
            t = np.linspace(0, 1, h).reshape(h, 1, 1)
            t = np.broadcast_to(t, (h, w, 1))
        else:
            tx = np.linspace(0, 1, w).reshape(1, w)
            ty = np.linspace(0, 1, h).reshape(h, 1)
            t = ((tx + ty) / 2.0).reshape(h, w, 1)
            t = np.broadcast_to(t, (h, w, 1))
        return (1 - t) * c1 + t * c2

    def _radial_gradient(self, w: int, h: int) -> np.ndarray:
        """Radial gradient from random center."""
        c1 = np.array(random_color(self.rng), dtype=np.float64)
        c2 = np.array(random_color(self.rng), dtype=np.float64)
        cx = w // 2 + self.rng.randint(-w // 4, w // 4)
        cy = h // 2 + self.rng.randint(-h // 4, h // 4)
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        t = (dist / (dist.max() + 1e-8)).reshape(h, w, 1)
        return (1 - t) * c1 + t * c2

    def _perlin_texture(self, w: int, h: int) -> np.ndarray:
        """Perlin noise mapped to random color palette."""
        scale = self.rng.uniform(20.0, 80.0)
        noise = perlin_noise(w, h, scale=scale, seed=int(self.rng.randint(0, 100000)))
        c1 = np.array(random_color(self.rng), dtype=np.float64)
        c2 = np.array(random_color(self.rng), dtype=np.float64)
        t = noise.reshape(h, w, 1)
        return (1 - t) * c1 + t * c2

    def _geometric_pattern(self, w: int, h: int) -> np.ndarray:
        """Random shapes simulating scene clutter."""
        bg = np.full((h, w, 3), random_color(self.rng), dtype=np.uint8)
        for _ in range(self.rng.randint(5, 25)):
            color = random_color(self.rng)
            st = self.rng.choice(["rect", "circle", "line", "ellipse"])
            if st == "rect":
                x1, y1 = self.rng.randint(0, w), self.rng.randint(0, h)
                x2 = x1 + self.rng.randint(20, max(21, w // 2))
                y2 = y1 + self.rng.randint(20, max(21, h // 2))
                cv2.rectangle(bg, (x1, y1), (x2, y2), color, self.rng.choice([-1, 1, 2, 3]))
            elif st == "circle":
                cx, cy = self.rng.randint(0, w), self.rng.randint(0, h)
                r = self.rng.randint(10, max(11, min(w, h) // 3))
                cv2.circle(bg, (cx, cy), r, color, self.rng.choice([-1, 1, 2]))
            elif st == "ellipse":
                cx, cy = self.rng.randint(0, w), self.rng.randint(0, h)
                ax1 = self.rng.randint(10, max(11, w // 3))
                ax2 = self.rng.randint(10, max(11, h // 3))
                angle = self.rng.randint(0, 180)
                cv2.ellipse(bg, (cx, cy), (ax1, ax2), angle, 0, 360, color, self.rng.choice([-1, 1, 2]))
            else:
                x1, y1 = self.rng.randint(0, w), self.rng.randint(0, h)
                x2, y2 = self.rng.randint(0, w), self.rng.randint(0, h)
                cv2.line(bg, (x1, y1), (x2, y2), color, self.rng.randint(1, 5))
        return bg.astype(np.float64)

    def _paper_grain(self, w: int, h: int) -> np.ndarray:
        """Paper-like texture with grain and stains."""
        base_choices = [(245, 235, 220), (240, 230, 210), (250, 245, 230),
                        (230, 220, 200), (255, 250, 240), (220, 210, 190)]
        base = np.array(base_choices[self.rng.randint(0, len(base_choices))], dtype=np.float64)
        img = np.full((h, w, 3), base, dtype=np.float64)
        img += self.rng.normal(0, self.rng.uniform(3, 12), (h, w, 3))
        for _ in range(self.rng.randint(0, 4)):
            cx, cy = self.rng.randint(0, w), self.rng.randint(0, h)
            radius = self.rng.randint(20, max(21, min(w, h) // 2))
            Y, X = np.ogrid[:h, :w]
            mask = ((X - cx)**2 + (Y - cy)**2) < radius**2
            img[mask] -= self.rng.uniform(10, 30)
        return img

    def _cloudy_sky(self, w: int, h: int) -> np.ndarray:
        """Multi-octave Perlin for cloud-like sky."""
        img = np.zeros((h, w, 3), dtype=np.float64)
        base_seed = int(self.rng.randint(0, 100000))
        for i, scale in enumerate([30, 60, 120]):
            n = perlin_noise(w, h, scale=scale, seed=base_seed + i)
            img += n.reshape(h, w, 1) * (200 / (i + 1))
        tint = np.array(random_color(self.rng), dtype=np.float64) / 255.0
        img = img * tint + 40
        return img

    def _stripe_pattern(self, w: int, h: int) -> np.ndarray:
        """Stripes with noise."""
        c1 = np.array(random_color(self.rng), dtype=np.float64)
        c2 = np.array(random_color(self.rng), dtype=np.float64)
        sw = self.rng.randint(5, 30)
        vertical = self.rng.choice([True, False])
        img = np.zeros((h, w, 3), dtype=np.float64)
        if vertical:
            for x in range(0, w, sw):
                img[:, x:x + sw] = c1 if (x // sw) % 2 == 0 else c2
        else:
            for y in range(0, h, sw):
                img[y:y + sw, :] = c1 if (y // sw) % 2 == 0 else c2
        img += self.rng.normal(0, 3, (h, w, 3))
        return img

    # ------------------------------------------------------------------
    # WEATHER / SCENE BACKGROUNDS
    # ------------------------------------------------------------------

    def _rainy_scene(self, w: int, h: int) -> np.ndarray:
        """Rainy scene: dark/grey background + rain streaks + water drops."""
        base_r = self.rng.randint(40, 90)
        base_g = self.rng.randint(50, 100)
        base_b = self.rng.randint(70, 130)
        noise = perlin_noise(w, h, scale=self.rng.uniform(40, 80),
                             seed=int(self.rng.randint(0, 100000)))
        img = np.zeros((h, w, 3), dtype=np.float64)
        img[:, :, 0] = base_r + noise * 30
        img[:, :, 1] = base_g + noise * 25
        img[:, :, 2] = base_b + noise * 20

        # Vertical rain streaks
        rain_layer = np.zeros((h, w), dtype=np.float64)
        n_streaks = self.rng.randint(50, 200)
        for _ in range(n_streaks):
            x = self.rng.randint(0, w)
            y_start = self.rng.randint(0, h)
            length = self.rng.randint(10, max(11, h // 2))
            thickness = self.rng.choice([1, 1, 1, 2])
            brightness = self.rng.uniform(0.3, 0.8)
            y_end = min(h, y_start + length)
            angle_offset = self.rng.randint(-3, 4)
            for dy in range(y_end - y_start):
                px = x + (dy * angle_offset) // max(1, length)
                if 0 <= px < w and 0 <= y_start + dy < h:
                    rain_layer[y_start + dy, max(0, px):min(w, px + thickness)] = brightness

        img[:, :, 0] += rain_layer * 180
        img[:, :, 1] += rain_layer * 190
        img[:, :, 2] += rain_layer * 200

        # Water droplets
        img_u8 = np.clip(img, 0, 255).astype(np.uint8)
        for _ in range(self.rng.randint(5, 30)):
            cx, cy = self.rng.randint(0, w), self.rng.randint(0, h)
            r = self.rng.randint(2, 8)
            bright = self.rng.randint(150, 230)
            cv2.circle(img_u8, (cx, cy), r, (bright, bright + 10, bright + 20), 1)
        return img_u8.astype(np.float64)

    def _sunny_scene(self, w: int, h: int) -> np.ndarray:
        """Sunny outdoor scene: bright warm tones + lens flare."""
        base = np.zeros((h, w, 3), dtype=np.float64)
        for y in range(h):
            t = y / max(1, h - 1)
            base[y, :, 0] = 180 + t * 60
            base[y, :, 1] = 200 + t * 40
            base[y, :, 2] = 240 - t * 80

        noise = perlin_noise(w, h, scale=self.rng.uniform(50, 100),
                             seed=int(self.rng.randint(0, 100000)))
        base += noise.reshape(h, w, 1) * 30

        # Sun flare
        sun_x = self.rng.randint(w // 4, 3 * w // 4)
        sun_y = self.rng.randint(0, max(1, h // 3))
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - sun_x)**2 + (Y - sun_y)**2)
        flare_radius = self.rng.uniform(30, 80)
        flare = np.exp(-dist**2 / (2 * flare_radius**2))
        base[:, :, 0] += flare * 80
        base[:, :, 1] += flare * 70
        base[:, :, 2] += flare * 40

        # Light rays
        img_u8 = np.clip(base, 0, 255).astype(np.uint8)
        for _ in range(self.rng.randint(3, 8)):
            angle = self.rng.uniform(0.3, 2.8)
            length = self.rng.randint(50, max(51, max(w, h)))
            ex = int(sun_x + length * np.cos(angle))
            ey = int(sun_y + length * np.sin(angle))
            cv2.line(img_u8, (sun_x, sun_y), (ex, ey),
                     (255, 245, 200), self.rng.randint(1, 3))
        img_u8 = cv2.GaussianBlur(img_u8, (5, 5), 0)
        return img_u8.astype(np.float64)

    def _sunset_sky(self, w: int, h: int) -> np.ndarray:
        """Sunset: orange-red-purple gradient sky."""
        img = np.zeros((h, w, 3), dtype=np.float64)
        colors = [
            np.array([30, 10, 60], dtype=np.float64),
            np.array([180, 50, 80], dtype=np.float64),
            np.array([255, 140, 50], dtype=np.float64),
            np.array([255, 200, 80], dtype=np.float64),
            np.array([200, 160, 100], dtype=np.float64),
        ]
        n_bands = len(colors)
        band_h = h / (n_bands - 1)
        for y in range(h):
            band = min(int(y / band_h), n_bands - 2)
            t = (y - band * band_h) / band_h
            img[y] = (1 - t) * colors[band] + t * colors[band + 1]
        noise = perlin_noise(w, h, scale=self.rng.uniform(40, 80),
                             seed=int(self.rng.randint(0, 100000)))
        cloud_mask = cv2.GaussianBlur((noise > 0.5).astype(np.float32), (15, 15), 0)
        img += cloud_mask.reshape(h, w, 1) * np.array([40, 20, -10], dtype=np.float64)
        return img

    def _night_scene(self, w: int, h: int) -> np.ndarray:
        """Night scene with stars and city lights."""
        base_r = self.rng.randint(5, 25)
        base_g = self.rng.randint(5, 20)
        base_b = self.rng.randint(15, 45)
        img = np.full((h, w, 3), [base_r, base_g, base_b], dtype=np.float64)
        noise = perlin_noise(w, h, scale=self.rng.uniform(30, 70),
                             seed=int(self.rng.randint(0, 100000)))
        img += noise.reshape(h, w, 1) * 15

        img_u8 = np.clip(img, 0, 255).astype(np.uint8)
        # Stars
        for _ in range(self.rng.randint(20, 100)):
            sx, sy = self.rng.randint(0, w), self.rng.randint(0, h)
            b = self.rng.randint(150, 255)
            cv2.circle(img_u8, (sx, sy), self.rng.choice([1, 1, 1, 2]), (b, b, b - 20), -1)
        # City lights
        for _ in range(self.rng.randint(3, 12)):
            lx = self.rng.randint(0, w)
            ly = self.rng.randint(max(1, h * 2 // 3), h)
            lr = self.rng.randint(5, 25)
            color = (self.rng.randint(200, 255), self.rng.randint(180, 255),
                     self.rng.randint(100, 200))
            cv2.circle(img_u8, (lx, ly), lr, color, -1)
        img_u8 = cv2.GaussianBlur(img_u8, (7, 7), 0)
        return img_u8.astype(np.float64)

    # ------------------------------------------------------------------
    # SURFACE / MATERIAL BACKGROUNDS
    # ------------------------------------------------------------------

    def _brick_wall(self, w: int, h: int) -> np.ndarray:
        """Brick wall texture."""
        brick_colors = [(165, 85, 55), (180, 95, 60), (150, 75, 50),
                        (190, 110, 70), (140, 70, 45), (170, 90, 55)]
        mortar_color = np.array([200, 195, 180], dtype=np.float64)
        brick_w = self.rng.randint(30, 60)
        brick_h = self.rng.randint(15, 30)
        mortar_t = self.rng.randint(2, 5)
        img = np.full((h, w, 3), mortar_color, dtype=np.float64)
        row = 0
        brick_row = 0
        while row < h:
            col = -brick_w // 2 if brick_row % 2 == 1 else 0
            while col < w:
                bc = np.array(brick_colors[self.rng.randint(0, len(brick_colors))],
                              dtype=np.float64) + self.rng.normal(0, 8, 3)
                x1, y1 = max(0, col + mortar_t), max(0, row + mortar_t)
                x2, y2 = min(w, col + brick_w), min(h, row + brick_h)
                if x2 > x1 and y2 > y1:
                    img[y1:y2, x1:x2] = bc
                    img[y1:y2, x1:x2] += self.rng.normal(0, 4, (y2 - y1, x2 - x1, 3))
                col += brick_w + mortar_t
            row += brick_h + mortar_t
            brick_row += 1
        return img

    def _wood_texture(self, w: int, h: int) -> np.ndarray:
        """Wood grain texture."""
        base = np.array([160, 110, 60], dtype=np.float64) + self.rng.normal(0, 15, 3)
        ring_color = np.array([120, 75, 35], dtype=np.float64) + self.rng.normal(0, 10, 3)
        noise = perlin_noise(w, h, scale=self.rng.uniform(15, 40),
                             seed=int(self.rng.randint(0, 100000)))
        Y, X = np.mgrid[:h, :w]
        freq = self.rng.uniform(0.02, 0.06)
        rings = np.sin((X * freq + noise * 10)) * 0.5 + 0.5
        t = rings.reshape(h, w, 1)
        img = (1 - t) * base + t * ring_color
        img += self.rng.normal(0, 5, (h, w, 3))
        return img

    def _metallic_surface(self, w: int, h: int) -> np.ndarray:
        """Brushed metallic surface."""
        base_val = self.rng.randint(140, 200)
        img = np.full((h, w, 3), base_val, dtype=np.float64)
        for y in range(h):
            img[y, :] += self.rng.normal(0, 8)
        noise = perlin_noise(w, h, scale=self.rng.uniform(20, 60),
                             seed=int(self.rng.randint(0, 100000)))
        img += noise.reshape(h, w, 1) * 25
        img_u8 = np.clip(img, 0, 255).astype(np.uint8)
        for _ in range(self.rng.randint(0, 8)):
            x1, y1 = self.rng.randint(0, w), self.rng.randint(0, h)
            x2, y2 = x1 + self.rng.randint(-50, 50), y1 + self.rng.randint(-20, 20)
            b = self.rng.randint(180, 240)
            cv2.line(img_u8, (x1, y1), (x2, y2), (b, b, b), 1)
        tint = 0.85 + 0.15 * np.array(random_color(self.rng), dtype=np.float64) / 255.0
        return img_u8.astype(np.float64) * tint

    def _concrete_wall(self, w: int, h: int) -> np.ndarray:
        """Concrete/cement wall texture."""
        base_val = self.rng.randint(150, 200)
        img = np.full((h, w, 3), base_val, dtype=np.float64)
        for scale in [15, 30, 60]:
            noise = perlin_noise(w, h, scale=scale, seed=int(self.rng.randint(0, 100000)))
            img += noise.reshape(h, w, 1) * self.rng.uniform(8, 20)
        img += self.rng.normal(0, 6, (h, w, 3))
        for _ in range(self.rng.randint(0, 5)):
            cx, cy = self.rng.randint(0, w), self.rng.randint(0, h)
            radius = self.rng.randint(15, max(16, min(w, h) // 3))
            Y, X = np.ogrid[:h, :w]
            dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
            mask = np.exp(-dist**2 / (2 * (radius * 0.5)**2))
            img -= mask.reshape(h, w, 1) * self.rng.uniform(20, 50)
        if self.rng.random() < 0.5:
            img[:, :, 0] += self.rng.uniform(-5, 10)
            img[:, :, 2] -= self.rng.uniform(0, 8)
        return img

    def _fabric_texture(self, w: int, h: int) -> np.ndarray:
        """Woven fabric / cloth texture."""
        base_color = np.array(random_color(self.rng), dtype=np.float64)
        img = np.full((h, w, 3), base_color, dtype=np.float64)
        thread_size = self.rng.randint(2, 6)
        weave_dark = self.rng.uniform(10, 25)
        for y in range(0, h, thread_size * 2):
            img[y:y + thread_size, :] -= weave_dark * 0.5
        for x in range(0, w, thread_size * 2):
            img[:, x:x + thread_size] -= weave_dark * 0.5
        img += self.rng.normal(0, 4, (h, w, 3))
        if self.rng.random() < 0.4:
            fold_x = self.rng.randint(w // 4, 3 * w // 4)
            spread = self.rng.randint(10, 30)
            X = np.arange(w).reshape(1, w)
            fold_mask = np.exp(-((X - fold_x)**2) / (2 * spread**2))
            img -= fold_mask.reshape(1, w, 1) * 25
        return img

    def _water_surface(self, w: int, h: int) -> np.ndarray:
        """Water surface with ripples and reflections."""
        base = np.array([40, 100, 140], dtype=np.float64) + self.rng.normal(0, 15, 3)
        img = np.full((h, w, 3), base, dtype=np.float64)
        Y, X = np.mgrid[:h, :w]
        for _ in range(self.rng.randint(3, 7)):
            fx = self.rng.uniform(0.01, 0.05)
            fy = self.rng.uniform(0.01, 0.05)
            phase = self.rng.uniform(0, 2 * np.pi)
            amp = self.rng.uniform(8, 25)
            ripple = np.sin(X * fx + Y * fy + phase) * amp
            img += ripple.reshape(h, w, 1) * np.array([0.3, 0.6, 1.0])
        noise = perlin_noise(w, h, scale=self.rng.uniform(30, 70),
                             seed=int(self.rng.randint(0, 100000)))
        img += noise.reshape(h, w, 1) * 20
        img_u8 = np.clip(img, 0, 255).astype(np.uint8)
        for _ in range(self.rng.randint(3, 10)):
            hx, hy = self.rng.randint(0, w), self.rng.randint(0, h)
            cv2.circle(img_u8, (hx, hy), self.rng.randint(2, 8), (220, 230, 245), -1)
        img_u8 = cv2.GaussianBlur(img_u8, (3, 3), 0)
        return img_u8.astype(np.float64)

    def _foliage_canopy(self, w: int, h: int) -> np.ndarray:
        """Dense foliage / leaf pattern."""
        base = np.array([30, 80, 25], dtype=np.float64) + self.rng.normal(0, 10, 3)
        img = np.full((h, w, 3), base, dtype=np.float64)
        for i, scale in enumerate([15, 30, 60]):
            noise = perlin_noise(w, h, scale=scale, seed=int(self.rng.randint(0, 100000)))
            img += noise.reshape(h, w, 1) * np.array([10, 40, 5], dtype=np.float64) / (i + 1)
        img_u8 = np.clip(img, 0, 255).astype(np.uint8)
        for _ in range(self.rng.randint(30, 80)):
            cx, cy = self.rng.randint(0, w), self.rng.randint(0, h)
            r = self.rng.randint(3, 15)
            color = (self.rng.randint(20, 60), self.rng.randint(60, 160), self.rng.randint(10, 50))
            cv2.circle(img_u8, (cx, cy), r, color, -1)
        for _ in range(self.rng.randint(5, 20)):
            dx, dy = self.rng.randint(0, w), self.rng.randint(0, h)
            cv2.circle(img_u8, (dx, dy), self.rng.randint(5, 20), (180, 200, 120), -1)
        img_u8 = cv2.GaussianBlur(img_u8, (5, 5), 0)
        return img_u8.astype(np.float64)

    def _signboard(self, w: int, h: int) -> np.ndarray:
        """Signboard / banner with border."""
        sign_colors = [
            (255, 255, 240), (240, 240, 255), (200, 230, 200),
            (255, 240, 200), (220, 220, 240), (250, 250, 220),
            (50, 80, 120), (120, 40, 40), (40, 100, 60),
            (255, 255, 255), (30, 30, 80),
        ]
        base = np.array(sign_colors[self.rng.randint(0, len(sign_colors))], dtype=np.float64)
        img = np.full((h, w, 3), base, dtype=np.float64)
        img += self.rng.normal(0, 3, (h, w, 3))
        bw = self.rng.randint(3, 10)
        bc = np.array(random_color(self.rng), dtype=np.float64)
        img[:bw, :] = bc; img[-bw:, :] = bc; img[:, :bw] = bc; img[:, -bw:] = bc
        if self.rng.random() < 0.5:
            fade = np.linspace(0.7, 1.0, min(15, max(1, min(h, w) // 4 - bw)))
            for i, f in enumerate(fade):
                idx = bw + i
                if idx < h: img[idx, :] *= f
                if idx < w: img[:, idx] *= f
        return img

    def _bokeh_blur(self, w: int, h: int) -> np.ndarray:
        """Out-of-focus bokeh circles (shallow depth of field)."""
        base = np.array(random_color(self.rng), dtype=np.float64) * 0.3 + 20
        img = np.full((h, w, 3), base, dtype=np.float64)
        noise = perlin_noise(w, h, scale=self.rng.uniform(40, 80),
                             seed=int(self.rng.randint(0, 100000)))
        img += noise.reshape(h, w, 1) * 30
        img_u8 = np.clip(img, 0, 255).astype(np.uint8)
        overlay = img_u8.copy()
        for _ in range(self.rng.randint(10, 40)):
            cx, cy = self.rng.randint(0, w), self.rng.randint(0, h)
            r = self.rng.randint(8, max(9, min(w, h) // 4))
            color = (self.rng.randint(100, 255), self.rng.randint(100, 255),
                     self.rng.randint(100, 255))
            cv2.circle(overlay, (cx, cy), r, color, -1)
        alpha = self.rng.uniform(0.2, 0.5)
        img_u8 = cv2.addWeighted(overlay, alpha, img_u8, 1 - alpha, 0)
        img_u8 = cv2.GaussianBlur(img_u8, (11, 11), 0)
        return img_u8.astype(np.float64)

    # ------------------------------------------------------------------
    # COMPOSITE
    # ------------------------------------------------------------------

    def _composite(self, w: int, h: int) -> np.ndarray:
        """Blend 2 sub-generators for complex backgrounds."""
        safe_gens = [g for g in self._generators if g != self._composite]
        idxs = self.rng.choice(len(safe_gens), size=2, replace=False)
        img1 = safe_gens[idxs[0]](w, h)
        img2 = safe_gens[idxs[1]](w, h)
        alpha = self.rng.uniform(0.3, 0.7)
        return alpha * img1 + (1 - alpha) * img2
