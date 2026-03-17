"""
Annotation writer — saves COCO-Text-style JSON annotations
and optional per-image sidecar label files.
"""
import json
import os
from typing import List, Dict, Any
from datetime import datetime


class AnnotationWriter:
    """Accumulates annotations and writes them in COCO-Text JSON format."""

    def __init__(self):
        self.images: List[Dict[str, Any]] = []
        self.annotations: List[Dict[str, Any]] = []
        self._ann_id = 0

    def add_image(self, image_id: int, file_name: str,
                  width: int, height: int,
                  word_annotations: List[Dict]):
        """
        Register one generated image and its text annotations.
        word_annotations: list of {"bbox": [x,y,w,h], "text": str, "language": str}
        """
        self.images.append({
            "id": image_id,
            "file_name": file_name,
            "width": width,
            "height": height,
        })

        for ann in word_annotations:
            self._ann_id += 1
            x, y, w, h = ann["bbox"]
            self.annotations.append({
                "id": self._ann_id,
                "image_id": image_id,
                "bbox": [x, y, w, h],
                "area": w * h,
                "utf8_string": ann["text"],
                "language": ann.get("language", "hi"),
                "script": "Devanagari",
                "legibility": "legible",
                "segmentation": [[x, y, x + w, y, x + w, y + h, x, y + h]],
            })

    def write_json(self, output_path: str):
        """Write all accumulated annotations as a single JSON file."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        data = {
            "info": {
                "description": "Synthetic Hindi Scene Text Dataset",
                "version": "1.0",
                "date_created": datetime.now().isoformat(),
                "script": "Devanagari",
                "language": "Hindi",
                "generator": "Indic Scene Text Generation Pipeline",
            },
            "images": self.images,
            "annotations": self.annotations,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[AnnotationWriter] Saved {len(self.annotations)} annotations "
              f"for {len(self.images)} images → {output_path}")

    @staticmethod
    def write_sidecar(labels_dir: str, image_filename: str, texts: List[str]):
        """Write a per-image sidecar .txt file with the label text."""
        os.makedirs(labels_dir, exist_ok=True)
        base = os.path.splitext(image_filename)[0]
        label_path = os.path.join(labels_dir, base + ".txt")
        with open(label_path, "w", encoding="utf-8") as f:
            for text in texts:
                f.write(text + "\n")
