#!/usr/bin/env python3
"""Generate dummy PNG images for CI testing"""
from pathlib import Path
from PIL import Image
import numpy as np

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DUMMY_IMAGE_DIR = BASE_DIR / "data" / "dummy_images"

# Image categories
CATEGORIES = ["NORMAL", "COVID", "VIRAL PNEUMONIA"]


def create_dummy_image(output_path, size=(224, 224)):
    """Create a valid dummy PNG image"""
    # Create a random RGB image
    img_array = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)
    img = Image.fromarray(img_array, mode='RGB')
    img.save(output_path, 'PNG')
    print(f"Created: {output_path}")


def main():
    """Generate dummy images for all categories"""
    for category in CATEGORIES:
        category_dir = DUMMY_IMAGE_DIR / category
        category_dir.mkdir(parents=True, exist_ok=True)

        # Create 3 dummy images per category
        for i in range(1, 4):
            image_path = category_dir / f"dummy_{i}.png"
            create_dummy_image(image_path)

    print(f"\nSuccessfully generated dummy images in {DUMMY_IMAGE_DIR}")


if __name__ == "__main__":
    main()
