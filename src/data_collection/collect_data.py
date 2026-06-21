"""
SignSpeakAI - Data Collection Module

Captures hand gesture images from webcam using MediaPipe hand detection.
Saves cropped hand ROI images organized by class label for training.

Usage:
    python -m src.data_collection.collect_data --label A --count 200
"""

import os
import sys
import cv2
import argparse
import time

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.hand_detector import HandDetector


def collect_data(label, output_dir="dataset/train", count=200, img_size=224):
    """
    Collect hand gesture images for a specific class label.

    Args:
        label: The class label (e.g., 'A', 'B', 'space', 'del').
        output_dir: Base output directory for training data.
        count: Number of images to collect.
        img_size: Size to resize saved images to.
    """
    # Create output directory
    save_dir = os.path.join(output_dir, label)
    os.makedirs(save_dir, exist_ok=True)

    # Count existing images to avoid overwriting
    existing = len([
        f for f in os.listdir(save_dir)
        if f.endswith(('.jpg', '.png', '.jpeg'))
    ])

    print(f"\n{'='*50}")
    print(f"  SignSpeakAI - Data Collection")
    print(f"{'='*50}")
    print(f"  Label:       {label}")
    print(f"  Save to:     {save_dir}")
    print(f"  Target:      {count} images")
    print(f"  Existing:    {existing} images")
    print(f"  Image size:  {img_size}x{img_size}")
    print(f"{'='*50}")
    print(f"\n  Controls:")
    print(f"    SPACE  → Start/Pause capture")
    print(f"    Q      → Quit")
    print(f"\n  Show your hand gesture and press SPACE to begin.\n")

    # Initialize components
    detector = HandDetector(
        min_detection_confidence=0.7,
        roi_padding=30
    )

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[Error] Cannot open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    capturing = False
    saved_count = 0
    img_index = existing

    try:
        while saved_count < count:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            annotated, roi, bbox = detector.detect(frame)

            # Status display
            status = "CAPTURING" if capturing else "PAUSED"
            color = (0, 255, 0) if capturing else (0, 165, 255)

            cv2.putText(
                annotated,
                f"Label: {label} | {status}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, color, 2
            )

            cv2.putText(
                annotated,
                f"Saved: {saved_count}/{count}",
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2
            )

            # Progress bar
            progress = saved_count / count
            bar_width = 300
            bar_x = 10
            bar_y = 80
            cv2.rectangle(annotated, (bar_x, bar_y), (bar_x + bar_width, bar_y + 15),
                          (50, 50, 50), -1)
            cv2.rectangle(annotated, (bar_x, bar_y),
                          (bar_x + int(bar_width * progress), bar_y + 15),
                          (0, 255, 0), -1)

            if capturing and roi is not None:
                # Resize and save
                resized = cv2.resize(roi, (img_size, img_size))
                filename = f"{label}_{img_index:05d}.jpg"
                filepath = os.path.join(save_dir, filename)

                cv2.imwrite(filepath, resized)
                saved_count += 1
                img_index += 1

                # Brief delay to avoid near-duplicate images
                time.sleep(0.05)

            # Show ROI preview
            if roi is not None:
                roi_preview = cv2.resize(roi, (150, 150))
                annotated[10:160, annotated.shape[1]-160:annotated.shape[1]-10] = roi_preview

            cv2.imshow("SignSpeakAI - Data Collection", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                capturing = not capturing
                state = "Started" if capturing else "Paused"
                print(f"  [{state}] capture for '{label}'")
            elif key == ord('q'):
                print("\n  Capture stopped by user.")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()

    print(f"\n  Done! Saved {saved_count} images for label '{label}'")
    print(f"  Total images in '{save_dir}': {img_index}")


def main():
    parser = argparse.ArgumentParser(
        description="SignSpeakAI - Collect hand gesture training data"
    )
    parser.add_argument(
        "--label", "-l",
        type=str,
        required=True,
        help="Class label to collect (e.g., A, B, space, del, nothing)"
    )
    parser.add_argument(
        "--count", "-c",
        type=int,
        default=200,
        help="Number of images to collect (default: 200)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="dataset/train",
        help="Output directory (default: dataset/train)"
    )
    parser.add_argument(
        "--size", "-s",
        type=int,
        default=224,
        help="Image size (default: 224)"
    )

    args = parser.parse_args()
    collect_data(
        label=args.label,
        output_dir=args.output,
        count=args.count,
        img_size=args.size
    )


if __name__ == "__main__":
    main()
