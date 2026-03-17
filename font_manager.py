"""
Font manager — discovers and manages Devanagari-capable fonts.
Downloads open-licensed fonts from Google Fonts if none are available locally.
"""
import os
import urllib.request
from typing import List, Tuple
from PIL import ImageFont

# Google Fonts direct download URLs (open-licensed Devanagari fonts)
GOOGLE_FONT_URLS = {
    "NotoSansDevanagari-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/notosansdevanagari/NotoSansDevanagari%5Bwdth%2Cwght%5D.ttf",
    "NotoSerifDevanagari-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/notoserifdevanagari/NotoSerifDevanagari%5Bwght%5D.ttf",
    "Hind-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/hind/Hind-Regular.ttf",
    "Hind-Bold.ttf": "https://github.com/google/fonts/raw/main/ofl/hind/Hind-Bold.ttf",
    "Hind-Light.ttf": "https://github.com/google/fonts/raw/main/ofl/hind/Hind-Light.ttf",
    "Tiro-Devanagari-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/tirodevanagarihindiregular/TiroDevanagariHindi-Regular.ttf",
    "Baloo2-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/baloo2/Baloo2%5Bwght%5D.ttf",
    "Kalam-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/kalam/Kalam-Regular.ttf",
}

# Windows system fonts that support Devanagari
SYSTEM_FONT_PATHS = [
    r"C:\Windows\Fonts\Nirmala.ttc",
    r"C:\Windows\Fonts\NirmalaB.ttc",
    r"C:\Windows\Fonts\mangal.ttf",
    r"C:\Windows\Fonts\MANGALB.TTF",
    r"C:\Windows\Fonts\aparaj.ttf",
    r"C:\Windows\Fonts\aparajb.ttf",
    r"C:\Windows\Fonts\kokila.ttf",
    r"C:\Windows\Fonts\kokilab.ttf",
]


class FontManager:
    """Manages a pool of Devanagari fonts for rendering."""

    def __init__(self, fonts_dir: str, font_size_range: Tuple[int, int] = (24, 72),
                 seed: int = 42):
        self.fonts_dir = fonts_dir
        self.font_size_range = font_size_range
        self.font_paths: List[str] = []
        self._rng = __import__("random").Random(seed)

        os.makedirs(self.fonts_dir, exist_ok=True)

        # Always try to download diverse fonts if fonts_dir is empty
        local_fonts = [f for f in os.listdir(self.fonts_dir)
                       if f.lower().endswith((".ttf", ".otf", ".ttc"))]
        if not local_fonts:
            print("[FontManager] No local fonts found. Downloading from Google Fonts...")
            self._download_fonts()

        self._discover_fonts()

        if not self.font_paths:
            raise RuntimeError(
                "No Devanagari fonts available. Place .ttf/.ttc files in the 'fonts/' directory."
            )

        print(f"[FontManager] {len(self.font_paths)} font(s) available.")

    def _discover_fonts(self):
        """Scan fonts_dir and system paths for usable fonts."""
        self.font_paths = []

        # Local fonts directory
        if os.path.isdir(self.fonts_dir):
            for fname in os.listdir(self.fonts_dir):
                if fname.lower().endswith((".ttf", ".otf", ".ttc")):
                    self.font_paths.append(os.path.join(self.fonts_dir, fname))

        # System fonts (fallback)
        for sp in SYSTEM_FONT_PATHS:
            if os.path.isfile(sp) and sp not in self.font_paths:
                self.font_paths.append(sp)

    def _download_fonts(self):
        """Download open-licensed Devanagari fonts from Google Fonts."""
        for fname, url in GOOGLE_FONT_URLS.items():
            dest = os.path.join(self.fonts_dir, fname)
            if os.path.isfile(dest):
                continue
            try:
                print(f"  Downloading {fname}...")
                urllib.request.urlretrieve(url, dest)
            except Exception as e:
                print(f"  WARNING: Failed to download {fname}: {e}")

    def get_random_font(self, size: int | None = None) -> ImageFont.FreeTypeFont:
        """Return a PIL ImageFont with random font face and size."""
        path = self._rng.choice(self.font_paths)
        if size is None:
            size = self._rng.randint(*self.font_size_range)
        try:
            font = ImageFont.truetype(path, size)
        except Exception:
            # Fallback to first working font
            for p in self.font_paths:
                try:
                    font = ImageFont.truetype(p, size)
                    return font
                except Exception:
                    continue
            raise RuntimeError("Cannot load any font.")
        return font

    def get_font_by_index(self, idx: int, size: int) -> ImageFont.FreeTypeFont:
        """Return a specific font by index."""
        path = self.font_paths[idx % len(self.font_paths)]
        return ImageFont.truetype(path, size)
