from pathlib import Path

# System & Weights Configuration
WEIGHTS_PATH = Path("weights/best.pt")
OUTPUT_DIR = Path("outputs")

# Confidence Threshold
# Set to 0.60 to eliminate false positives on dry/normal asphalt road surfaces (e.g. 0.56 false detection)
CONFIDENCE = 0.60

# Video Performance & Temporal Confirmation Settings
FRAME_SKIP = 2
REQUIRED_POSITIVE_FRAMES = 5
REQUIRED_NEGATIVE_FRAMES = 3

# File Extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
