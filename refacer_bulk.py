# refacer_bulk.py
#
# Example usage:
# python refacer_bulk.py --input_path ./input --dest_face myface.jpg --facetoreplace face1.jpg --threshold 0.3
#
# Or, to disable similarity check (i.e., just apply the destination face to all detected faces):
# python refacer_bulk.py --input_path ./input --dest_face myface.jpg

import argparse
import os
import cv2
from pathlib import Path
from refacer import Refacer
from PIL import Image
import time
import pyfiglet

def parse_args():
    parser = argparse.ArgumentParser(description="Bulk Image Refacer")
    parser.add_argument("--input_path", type=str, required=True, help="Directory containing input images")
    parser.add_argument("--dest_face", type=str, required=True, help="Path to destination face image")
    parser.add_argument("--facetoreplace", type=str, default=None, help="Path to face to replace (origin face)")
    parser.add_argument("--threshold", type=float, default=0.2, help="Similarity threshold (default: 0.2)")
    parser.add_argument("--force_cpu", action="store_true", help="Force CPU mode")
    parser.add_argument("--colab_performance", action="store_true", help="Enable Colab performance tweaks")
    return parser.parse_args()

def main():
    print("\033[94m" + pyfiglet.Figlet(font='slant').renderText("NeoRefacer") + "\033[0m")
    
    args = parse_args()

    input_dir = Path(args.input_path)

    refacer = Refacer(force_cpu=args.force_cpu, colab_performance=args.colab_performance)

    # Load destination and origin face
    dest_img = cv2.imread(args.dest_face)
    if dest_img is None:
        raise ValueError(f"Destination face image not found: {args.dest_face}")
    
    origin_img = None
    if args.facetoreplace:
        origin_img = cv2.imread(args.facetoreplace)
        if origin_img is None:
            raise ValueError(f"Face to replace image not found: {args.facetoreplace}")

    disable_similarity = origin_img is None

    faces_config = [{
        'origin': origin_img,
        'destination': dest_img,
        'threshold': args.threshold
    }]

    refacer.prepare_faces(faces_config, disable_similarity=disable_similarity)

    print(f"Processing images from: {input_dir}")
    image_files = list(input_dir.glob("*"))
    supported_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    for image_path in image_files:
        if image_path.suffix.lower() not in supported_exts:
            print(f"Skipping non-image file: {image_path}")
            continue

        print(f"Refacing: {image_path}")
        try:
            refaced_path = refacer.reface_image(str(image_path), faces_config, disable_similarity=disable_similarity)
            print(f"Saved to: {refaced_path}")
        except Exception as e:
            print(f"Failed to process {image_path}: {e}")

if __name__ == "__main__":
    main()
