from pathlib import Path
import shutil

# Define image categories and paths
LABELS = ["COVID", "NORMAL", "VIRAL PNEUMONIA"]
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "images"
DST_DIR = BASE_DIR / "data" / "dummy_images"
NUM_IMAGES_PER_CLASS = 3  # keep small for CI


def create_dummy_images():
    for label in LABELS:
        src_dir = DATA_DIR / label
        dst_dir = DST_DIR / label
        dst_dir.mkdir(parents=True, exist_ok=True)

        image_files = sorted([f for f in src_dir.glob("*") if f.is_file()])
        for i, img_path in enumerate(image_files[:NUM_IMAGES_PER_CLASS]):
            ext = img_path.suffix
            dummy_filename = f"dummy_{i + 1}{ext}"
            dst_path = dst_dir / dummy_filename
            shutil.copy(img_path, dst_path)

    print(f"✅ Dummy image copies created in: {DST_DIR}")


if __name__ == "__main__":
    create_dummy_images()
