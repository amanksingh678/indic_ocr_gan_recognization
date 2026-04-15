# 🔤 Indic OCR GAN Recognization

A deep learning pipeline for generating synthetic **Indic/Hindi text datasets** using GAN-based augmentation techniques, designed to train and evaluate OCR (Optical Character Recognition) models on Hindi and other Indic scripts.

---

## 📌 Overview

Training robust OCR models for Indic languages is challenging due to the lack of large, annotated datasets. This project solves that by **synthetically generating realistic Hindi text images** with varied backgrounds, fonts, and augmentations — enabling OCR models to generalize better to real-world conditions.

The pipeline produces annotated word-level image datasets (e.g., `output_dataset_word/hindi_clean_realbg_100k`) ready for training OCR/recognition models.

---

## 🗂️ Project Structure

```
indic_ocr_gan_recognization/
│
├── main.py                  # Entry point — runs the full pipeline
├── pipeline.py              # Orchestrates the generation workflow
├── config.py                # Configuration: paths, fonts, image size, etc.
│
├── corpus_loader.py         # Loads Hindi/Indic word corpus for text generation
├── font_manager.py          # Manages font selection and rendering
├── text_renderer.py         # Renders text onto images
├── background_generator.py  # Generates realistic backgrounds
├── augmentor.py             # Applies augmentations (blur, noise, distortion, etc.)
├── annotation_writer.py     # Writes ground-truth annotation files
│
├── quick_verify.py          # Quick sanity check on generated samples
├── verify_output.py         # Full verification of output dataset quality
├── utils.py                 # Helper utilities
│
└── output_dataset_word/
    └── hindi_clean_realbg_100k/   # Generated dataset (100k word images)
```

---

## ✨ Features

- 🖋️ **Multi-font support** for Hindi and Indic scripts
- 🌄 **Realistic background generation** for natural-looking text images
- 🔁 **GAN-inspired augmentation** for diverse training samples
- 📝 **Auto annotation writer** for ground-truth labels
- ✅ **Built-in verification tools** to validate dataset quality
- 📦 Generates large-scale datasets (100k+ samples)

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

> Python 3.8+ recommended

### Run the Pipeline

```bash
python main.py
```

To configure output path, fonts, image dimensions, and sample count, edit `config.py`.

### Verify Output

```bash
python quick_verify.py      # Fast check on a few samples
python verify_output.py     # Full dataset verification
```

---

## 📊 Dataset Output

The pipeline generates word-level cropped text images with corresponding annotation files:

| Property | Value |
|----------|-------|
| Script | Hindi (Devanagari) |
| Dataset size | ~100,000 word images |
| Background | Real + Synthetic |
| Annotation format | Text label per image |

---

## 👥 Contributors

- [amanksingh678](https://github.com/amanksingh678) — Aman Kumar Singh  
- [adityakumarsingh0701](https://github.com/adityakumarsingh0701)

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 📬 Contact

**Aman Kumar Singh** — amankumar.singh8501@gmail.com  
GitHub: [@amanksingh678](https://github.com/amanksingh678)

---

> *Built to solve real-world OCR challenges for underrepresented Indic languages.*
