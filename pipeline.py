"""
Pipeline orchestrator — ties all modules together.
Generates synthetic scene text images for Hindi OCR training.
"""
import os
import sys
import argparse
import time
import numpy as np
from tqdm import tqdm
from PIL import Image

from config import PipelineConfig
from corpus_loader import CorpusLoader
from font_manager import FontManager
from background_generator import BackgroundGenerator
from text_renderer import TextRenderer
from augmentor import Augmentor
from annotation_writer import AnnotationWriter


def generate_single_image(
    idx: int,
    config: PipelineConfig,
    corpus: CorpusLoader,
    font_mgr: FontManager,
    bg_gen: BackgroundGenerator,
    renderer: TextRenderer,
    augmentor: Augmentor,
) -> dict:
    """
    Generate one synthetic scene text image.
    Returns dict with image, filename, and annotations.
    """
    rng = np.random.RandomState(config.seed + idx)

    # Random image size
    img_w = rng.randint(*config.image_width_range)
    img_h = rng.randint(*config.image_height_range)

    # Generate background
    background = bg_gen.generate(img_w, img_h)

    # Sample text and render
    if config.generation_mode == "line":
        text = corpus.get_random_line()
        font = font_mgr.get_random_font()
        image, annotations = renderer.render_line(background, text, font,
                                                   config.text_color_auto_contrast)
    else:
        # Word mode: render 1-N words
        n_words = rng.randint(1, config.max_words_per_image + 1)
        words = [corpus.get_random_word() for _ in range(n_words)]
        image, annotations = renderer.render_words(
            background, words,
            font_fn=lambda: font_mgr.get_random_font(),
            auto_contrast=config.text_color_auto_contrast,
        )

    # Apply augmentations
    image, annotations = augmentor.augment(image, annotations)

    # Filename
    ext = config.image_format
    filename = f"scene_{idx:06d}.{ext}"

    return {
        "image": image,
        "filename": filename,
        "annotations": annotations,
        "image_id": idx,
    }


def run_pipeline(config: PipelineConfig):
    """Execute the full synthetic data generation pipeline."""
    print("=" * 60)
    print("  Indic Script-Aware Synthetic Scene Text Generator")
    print("=" * 60)
    print(f"  Mode           : {config.generation_mode}")
    print(f"  Images to gen  : {config.num_images}")
    print(f"  Output dir     : {config.output_dir}")
    print(f"  Image format   : {config.image_format}")
    print(f"  Seed           : {config.seed}")
    print("=" * 60)

    # Create output directories
    os.makedirs(config.images_dir, exist_ok=True)
    if config.save_sidecar_txt:
        os.makedirs(config.labels_dir, exist_ok=True)

    # Initialize modules
    print("\n[1/6] Loading corpus...")
    corpus = CorpusLoader(config.corpus_path, seed=config.seed)
    corpus.load()

    print("\n[2/6] Initializing font manager...")
    font_mgr = FontManager(config.fonts_dir, config.font_size_range, seed=config.seed)

    print("\n[3/6] Initializing background generator...")
    bg_gen = BackgroundGenerator(seed=config.seed)

    print("\n[4/6] Initializing text renderer...")
    renderer = TextRenderer(
        seed=config.seed,
        shadow_prob=config.text_shadow_prob,
        outline_prob=config.text_outline_prob,
        rotation_range=config.rotation_range,
    )

    print("\n[5/6] Initializing augmentor...")
    augmentor = Augmentor(config)

    print("\n[6/6] Generating images...\n")
    ann_writer = AnnotationWriter()

    start_time = time.time()
    success_count = 0
    error_count = 0

    for idx in tqdm(range(config.num_images), desc="Generating", unit="img"):
        try:
            result = generate_single_image(
                idx, config, corpus, font_mgr, bg_gen, renderer, augmentor
            )

            image = result["image"]
            filename = result["filename"]
            annotations = result["annotations"]
            image_id = result["image_id"]

            # Save image
            img_path = os.path.join(config.images_dir, filename)
            if config.image_format == "jpg":
                image.save(img_path, "JPEG", quality=95)
            else:
                image.save(img_path, "PNG")

            # Record annotations
            ann_writer.add_image(
                image_id=image_id,
                file_name=filename,
                width=image.size[0],
                height=image.size[1],
                word_annotations=annotations,
            )

            # Sidecar label file
            if config.save_sidecar_txt:
                texts = [a["text"] for a in annotations]
                AnnotationWriter.write_sidecar(config.labels_dir, filename, texts)

            success_count += 1

        except Exception as e:
            error_count += 1
            if error_count <= 5:
                print(f"\n  WARNING: Error generating image {idx}: {e}")
            elif error_count == 6:
                print(f"\n  (Suppressing further error messages...)")

    # Write master annotation file
    ann_writer.write_json(config.annotations_path)

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"  DONE!")
    print(f"  Generated : {success_count} images")
    print(f"  Errors    : {error_count}")
    print(f"  Time      : {elapsed:.1f}s ({success_count / max(elapsed, 0.1):.1f} img/s)")
    print(f"  Images    : {config.images_dir}")
    print(f"  Annotations: {config.annotations_path}")
    if config.save_sidecar_txt:
        print(f"  Labels    : {config.labels_dir}")
    print(f"{'=' * 60}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Synthetic Scene Text Generation Pipeline for Hindi OCR"
    )
    parser.add_argument("--num_images", "-n", type=int, default=1000,
                        help="Number of images to generate (default: 1000)")
    parser.add_argument("--mode", "-m", choices=["word", "line"], default="word",
                        help="Generation mode: 'word' or 'line' (default: word)")
    parser.add_argument("--output_dir", "-o", type=str, default="output",
                        help="Output directory (default: output)")
    parser.add_argument("--corpus", type=str,
                        default="hin-in_web_2015_1M-sentences.txt",
                        help="Path to corpus file")
    parser.add_argument("--fonts_dir", type=str, default="fonts",
                        help="Path to fonts directory")
    parser.add_argument("--image_format", type=str, choices=["jpg", "png"],
                        default="jpg", help="Output image format")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--min_font_size", type=int, default=24,
                        help="Minimum font size (default: 24)")
    parser.add_argument("--max_font_size", type=int, default=72,
                        help="Maximum font size (default: 72)")
    parser.add_argument("--no_sidecar", action="store_true",
                        help="Disable per-image .txt label files")
    return parser.parse_args()


def main():
    """Entry point."""
    args = parse_args()

    config = PipelineConfig(
        corpus_path=args.corpus,
        fonts_dir=args.fonts_dir,
        output_dir=args.output_dir,
        num_images=args.num_images,
        generation_mode=args.mode,
        image_format=args.image_format,
        seed=args.seed,
        font_size_range=(args.min_font_size, args.max_font_size),
        save_sidecar_txt=not args.no_sidecar,
    )

    run_pipeline(config)


if __name__ == "__main__":
    main()
