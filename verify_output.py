"""
Comprehensive verification of the 10,000-image generated dataset.
Checks: count, diversity (backgrounds, colors, sizes), annotation accuracy,
text-bbox alignment, and background type distribution.
"""
import json
import os
import sys
import numpy as np
from PIL import Image
from collections import Counter

OUTPUT_DIR = "output"
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
LABELS_DIR = os.path.join(OUTPUT_DIR, "labels")
ANN_FILE = os.path.join(OUTPUT_DIR, "annotations.json")


def main():
    print("=" * 70)
    print("  COMPREHENSIVE DATASET VERIFICATION")
    print("=" * 70)

    # ---- 1. Count check ----
    images = sorted([f for f in os.listdir(IMAGES_DIR) if f.endswith(('.jpg', '.png'))])
    labels = sorted([f for f in os.listdir(LABELS_DIR) if f.endswith('.txt')])
    print(f"\n[1] FILE COUNTS")
    print(f"  Images generated : {len(images)}")
    print(f"  Label files      : {len(labels)}")

    # ---- 2. Annotation check ----
    with open(ANN_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    ann_images = data["images"]
    annotations = data["annotations"]
    print(f"\n[2] ANNOTATIONS (COCO-Text Format)")
    print(f"  Images in JSON   : {len(ann_images)}")
    print(f"  Annotations      : {len(annotations)}")
    print(f"  Avg ann/image    : {len(annotations) / max(1, len(ann_images)):.1f}")
    print(f"  Script           : {data['info']['script']}")
    print(f"  Language         : {data['info']['language']}")

    # Verify all images have annotations
    img_ids_with_ann = set(a["image_id"] for a in annotations)
    img_ids_all = set(im["id"] for im in ann_images)
    missing = img_ids_all - img_ids_with_ann
    print(f"  Images with ann  : {len(img_ids_with_ann)}")
    print(f"  Missing ann      : {len(missing)}")

    # ---- 3. Size diversity ----
    print(f"\n[3] SIZE DIVERSITY")
    widths = [im["width"] for im in ann_images]
    heights = [im["height"] for im in ann_images]
    print(f"  Width  range     : {min(widths)} - {max(widths)}")
    print(f"  Height range     : {min(heights)} - {max(heights)}")
    print(f"  Unique sizes     : {len(set(zip(widths, heights)))}")

    # ---- 4. Color / Background diversity analysis ----
    print(f"\n[4] BACKGROUND DIVERSITY ANALYSIS (sampling 200 images)")
    sample_indices = np.linspace(0, len(images) - 1, 200, dtype=int)
    mean_colors = []
    std_devs = []
    luminances = []

    for idx in sample_indices:
        img = np.array(Image.open(os.path.join(IMAGES_DIR, images[idx])))
        mc = img.mean(axis=(0, 1))
        sd = img.std(axis=(0, 1)).mean()
        lum = 0.299 * mc[0] + 0.587 * mc[1] + 0.114 * mc[2]
        mean_colors.append(mc)
        std_devs.append(sd)
        luminances.append(lum)

    mean_colors = np.array(mean_colors)
    std_devs = np.array(std_devs)
    luminances = np.array(luminances)

    dark = (luminances < 80).sum()
    medium = ((luminances >= 80) & (luminances < 170)).sum()
    bright = (luminances >= 170).sum()
    simple = (std_devs < 15).sum()
    moderate = ((std_devs >= 15) & (std_devs < 40)).sum()
    complex_bg = (std_devs >= 40).sum()

    print(f"  Luminance distribution:")
    print(f"    Dark  (<80)    : {dark} ({dark/2:.0f}%)")
    print(f"    Medium (80-170): {medium} ({medium/2:.0f}%)")
    print(f"    Bright (>170)  : {bright} ({bright/2:.0f}%)")
    print(f"  Complexity distribution (pixel std dev):")
    print(f"    Simple (<15)   : {simple} ({simple/2:.0f}%)")
    print(f"    Moderate(15-40): {moderate} ({moderate/2:.0f}%)")
    print(f"    Complex (>40)  : {complex_bg} ({complex_bg/2:.0f}%)")
    print(f"  Mean R range     : {mean_colors[:,0].min():.0f} - {mean_colors[:,0].max():.0f}")
    print(f"  Mean G range     : {mean_colors[:,1].min():.0f} - {mean_colors[:,1].max():.0f}")
    print(f"  Mean B range     : {mean_colors[:,2].min():.0f} - {mean_colors[:,2].max():.0f}")
    print(f"  Avg pixel StdDev : {std_devs.mean():.1f} (higher=more textured)")

    # ---- 5. Text content diversity ----
    print(f"\n[5] TEXT CONTENT DIVERSITY")
    all_texts = [a["utf8_string"] for a in annotations]
    unique_texts = set(all_texts)
    avg_len = np.mean([len(t) for t in all_texts])
    print(f"  Total text boxes : {len(all_texts)}")
    print(f"  Unique texts     : {len(unique_texts)}")
    print(f"  Avg text length  : {avg_len:.1f} chars")
    print(f"  All Devanagari   : {all(a['script']=='Devanagari' for a in annotations)}")
    print(f"  All Hindi (hi)   : {all(a['language']=='hi' for a in annotations)}")

    # ---- 6. Bbox sanity ----
    print(f"\n[6] BOUNDING BOX SANITY CHECK")
    valid_bbox = 0
    invalid_bbox = 0
    for ann in annotations:
        x, y, bw, bh = ann["bbox"]
        img_entry = next((im for im in ann_images if im["id"] == ann["image_id"]), None)
        if img_entry:
            iw, ih = img_entry["width"], img_entry["height"]
            if x >= 0 and y >= 0 and bw > 0 and bh > 0 and x+bw <= iw+5 and y+bh <= ih+5:
                valid_bbox += 1
            else:
                invalid_bbox += 1
    print(f"  Valid bboxes     : {valid_bbox}")
    print(f"  Invalid bboxes   : {invalid_bbox}")
    print(f"  Validity rate    : {valid_bbox/max(1,valid_bbox+invalid_bbox)*100:.1f}%")

    # ---- 7. Samples ----
    print(f"\n[7] SAMPLE ANNOTATIONS (first 5)")
    for ann in annotations[:5]:
        print(f"  img_id={ann['image_id']}, bbox={ann['bbox']}, "
              f"text=\"{ann['utf8_string'][:40]}\"")

    # ---- 8. Sidecar labels ----
    print(f"\n[8] SIDECAR LABEL CHECK")
    if labels:
        with open(os.path.join(LABELS_DIR, labels[0]), "r", encoding="utf-8") as f:
            print(f"  Sample ({labels[0]}): {f.read().strip()[:80]}")
        matched = sum(1 for lf in labels[:100]
                      if any(im["file_name"] == os.path.splitext(lf)[0]+".jpg"
                             for im in ann_images))
        print(f"  Labels matching images (of first 100): {matched}")

    # ---- 9. Disk ----
    print(f"\n[9] DISK USAGE")
    total_size = sum(os.path.getsize(os.path.join(IMAGES_DIR, f)) for f in images)
    ann_size = os.path.getsize(ANN_FILE)
    print(f"  Total image size : {total_size/(1024*1024):.1f} MB")
    print(f"  Avg image size   : {total_size/max(1,len(images))/1024:.1f} KB")
    print(f"  Annotations file : {ann_size/(1024*1024):.1f} MB")

    # ---- 10. Verdict ----
    print(f"\n{'='*70}")
    issues = []
    if len(images) < 10000:
        issues.append(f"Only {len(images)} images (expected 10000)")
    if invalid_bbox > len(annotations)*0.05:
        issues.append(f"Too many invalid bboxes: {invalid_bbox}")
    if len(unique_texts) < 1000:
        issues.append(f"Low text diversity: {len(unique_texts)} unique")
    if dark < 5 and bright < 5:
        issues.append("Backgrounds lack luminance diversity")
    if complex_bg < 5:
        issues.append("Backgrounds lack texture complexity")
    if issues:
        print("  ISSUES FOUND:")
        for issue in issues:
            print(f"    WARNING: {issue}")
    else:
        print("  ALL CHECKS PASSED - Dataset is ready for OCR training!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
