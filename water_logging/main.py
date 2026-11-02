import os
import sys
import cv2
from pathlib import Path
from ultralytics import YOLO

import config


def check_model_weights():
    if not config.WEIGHTS_PATH.exists():
        print(f"ERROR: best.pt model file not found at {config.WEIGHTS_PATH}\n")
        print("Please place your trained YOLO model at:")
        print(f"  {config.WEIGHTS_PATH}")
        sys.exit(1)


def check_waterlogging_detection(result):
    """
    Check if waterlogging is detected in YOLO prediction result.
    Returns (has_detection, max_confidence).
    """
    if result is None or len(result) == 0:
        return False, 0.0

    boxes = getattr(result, "boxes", None)
    if boxes is not None and len(boxes) > 0:
        confidences = boxes.conf.cpu().numpy()
        if len(confidences) > 0:
            return True, float(confidences.max())

    masks = getattr(result, "masks", None)
    if masks is not None and len(masks) > 0:
        return True, 1.0

    return False, 0.0


def draw_overlay(image, is_detected, confidence=None):
    """Draw a clear status overlay on top-left of the image/frame."""
    h, w = image.shape[:2]
    color = (0, 0, 255) if is_detected else (0, 255, 0)  # Red if detected, Green if not
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.6, min(w, h) / 750.0)
    thickness = max(2, int(font_scale * 2.0))

    lines = []
    if is_detected:
        lines.append("WATERLOGGING DETECTED")
        if confidence is not None:
            lines.append(f"Confidence: {confidence:.2f}")
    else:
        lines.append("NO WATERLOGGING")

    line_sizes = [cv2.getTextSize(line, font, font_scale, thickness) for line in lines]
    max_w = max(size[0][0] for size in line_sizes)
    
    padding = 10
    spacing = 8
    rect_w = max_w + (2 * padding)
    rect_h = sum(size[0][1] for size in line_sizes) + (len(lines) - 1) * spacing + (2 * padding)

    x1, y1 = 15, 15
    x2, y2 = x1 + rect_w, y1 + rect_h

    # Semi-transparent dark background for readability
    overlay = image.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, image, 0.35, 0, image)

    # Outer border matching status color
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    # Text placement
    curr_y = y1 + padding
    for i, line in enumerate(lines):
        th = line_sizes[i][0][1]
        curr_y += th
        cv2.putText(image, line, (x1 + padding, curr_y), font, font_scale, color, thickness, cv2.LINE_AA)
        curr_y += spacing


def process_image(model, image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Unable to read image '{image_path}'.")
        return

    results = model.predict(source=image, conf=config.CONFIDENCE, verbose=False)
    result = results[0]

    has_detection, max_conf = check_waterlogging_detection(result)
    
    if has_detection:
        print(f"\n>>> WATERLOGGING DETECTED (Confidence: {max_conf:.2f}) <<<\n")
        annotated_image = result.plot()
        draw_overlay(annotated_image, is_detected=True, confidence=max_conf)
    else:
        print("\n>>> NO WATERLOGGING <<<\n")
        annotated_image = image.copy()
        draw_overlay(annotated_image, is_detected=False)

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_filename = f"result_{image_path.name}"
    output_path = config.OUTPUT_DIR / output_filename

    cv2.imwrite(str(output_path), annotated_image)
    print(f"Saved result to: {output_path}")


def process_video(model, video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Unable to open video '{video_path}'.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps != fps:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_filename = f"result_{video_path.stem}.mp4"
    output_path = config.OUTPUT_DIR / output_filename

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    print(f"\nProcessing video frames (skip={config.FRAME_SKIP}, pos_thresh={config.REQUIRED_POSITIVE_FRAMES}, neg_thresh={config.REQUIRED_NEGATIVE_FRAMES})...")

    # Temporal confirmation state
    pos_count = 0
    neg_count = 0
    is_active = False
    latest_conf = 0.0
    latest_result = None
    latest_raw_detection = False

    frame_count = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            is_processed_frame = (frame_count % config.FRAME_SKIP == 0) or (latest_result is None)

            if is_processed_frame:
                results = model.predict(source=frame, conf=config.CONFIDENCE, verbose=False)
                result = results[0]
                has_detection, max_conf = check_waterlogging_detection(result)
                
                latest_result = result
                latest_raw_detection = has_detection

                if has_detection:
                    pos_count += 1
                    neg_count = 0
                    latest_conf = max_conf
                    if pos_count >= config.REQUIRED_POSITIVE_FRAMES:
                        is_active = True
                else:
                    neg_count += 1
                    pos_count = 0
                    if neg_count >= config.REQUIRED_NEGATIVE_FRAMES:
                        is_active = False

            # Render output frame
            if is_active:
                if latest_raw_detection and latest_result is not None and len(latest_result) > 0:
                    annotated_frame = latest_result.plot(img=frame)
                else:
                    annotated_frame = frame.copy()
                draw_overlay(annotated_frame, is_detected=True, confidence=latest_conf)
            else:
                annotated_frame = frame.copy()
                draw_overlay(annotated_frame, is_detected=False)

            out.write(annotated_frame)

            if total_frames > 0 and frame_count % 30 == 0:
                print(f"Processed {frame_count}/{total_frames} frames...")

    finally:
        cap.release()
        out.release()

    if is_active:
        print("\n>>> WATERLOGGING DETECTED IN VIDEO <<<\n")
    else:
        print("\n>>> NO WATERLOGGING DETECTED IN VIDEO <<<\n")

    print(f"Saved result to: {output_path}")


def main():
    check_model_weights()

    print("=" * 45)
    print("       WATERLOGGING DETECTION SYSTEM")
    print("=" * 45)

    try:
        model = YOLO(str(config.WEIGHTS_PATH))
        print(f"Model loaded successfully. Type: YOLO ({model.task})")
    except Exception as e:
        print(f"Error loading model from {config.WEIGHTS_PATH}: {e}")
        sys.exit(1)

    while True:
        try:
            user_input = input("\nEnter image/video path (or type 'exit'): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting system.")
            break

        if not user_input:
            continue

        clean_path = user_input.strip('"\'')
        if clean_path.lower() == 'exit':
            print("Exiting Waterlogging Detection System. Goodbye!")
            break

        file_path = Path(clean_path)
        if not file_path.exists() or not file_path.is_file():
            print("File not found. Please enter a valid path.")
            continue

        ext = file_path.suffix.lower()
        if ext in config.IMAGE_EXTENSIONS:
            process_image(model, file_path)
        elif ext in config.VIDEO_EXTENSIONS:
            process_video(model, file_path)
        else:
            print(f"Unsupported file format '{ext}'.")
            print("Supported images:", ", ".join(sorted(config.IMAGE_EXTENSIONS)))
            print("Supported videos:", ", ".join(sorted(config.VIDEO_EXTENSIONS)))


if __name__ == "__main__":
    main()
