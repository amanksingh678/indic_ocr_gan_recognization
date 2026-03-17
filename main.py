"""
Main entry point for the Indic Script-Aware Synthetic Scene Text Generation Pipeline.

Usage:
    python main.py                          # Generate 1000 word-mode images
    python main.py -n 5000 -m line          # Generate 5000 line-mode images
    python main.py -n 100 -m word -o out    # 100 word images to 'out/' folder
    python main.py --help                   # Show all options
"""
from pipeline import main

if __name__ == "__main__":
    main()

