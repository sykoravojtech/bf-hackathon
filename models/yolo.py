import argparse
import os
import time

import cv2
import numpy as np
from ultralytics import YOLO


class YOLODetector:
    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.4, device="auto"):
        """
        Initialize the YOLO-based human detector

        Args:
            model_path: Path to YOLOv8 model or model name to download
            conf_threshold: Confidence threshold for detections (0.0-1.0)
            device: Device to run inference on ('cpu', 'cuda:0', or 'auto' for automatic selection)
        """
        self.conf_threshold = conf_threshold

        # Load the YOLOv8 model
        try:
            self.model = YOLO(model_path)
            print(f"YOLOv8 model loaded: {model_path}")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            raise

        # Set device for inference
        self.device = device

    def detect_person(self, frame):
        """
        Detect people in the given frame

        Args:
            frame: Input image/frame

        Returns:
            List of bounding boxes in format [x1, y1, x2, y2]
        """
        bboxs = []

        # Run YOLOv8 inference
        results = self.model(
            frame, conf=self.conf_threshold, classes=0
        )  # Class 0 is 'person' in COCO dataset

        # Process detection results
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get box coordinates (convert to int for drawing)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Get confidence score
                conf = float(box.conf[0])

                # Add to bounding boxes list
                bboxs.append([x1, y1, x2, y2])

                # Draw rectangle and confidence score
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)
                score = int(conf * 100)
                cv2.putText(
                    frame,
                    f"Person {score}%",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )

        return bboxs

    def process_video(
        self, video_path, output_path=None, display=True, close_threshold=0.15
    ):
        """
        Process a video file, detect persons in each frame and optionally save output

        Args:
            video_path: Path to input video file
            output_path: Path to save output video (None to skip saving)
            display: Whether to display video while processing
            close_threshold: Threshold for determining if a human is close

        Returns:
            List of frames where humans were detected as close
        """
        # Check if video file exists
        if not os.path.isfile(video_path):
            print(f"Error: Video file '{video_path}' does not exist")
            return []

        # Open the video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video '{video_path}'")
            return []

        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        img_size = (frame_width, frame_height)

        # Initialize video writer if output path is provided
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        frames_with_close_humans = []
        frame_count = 0

        print(f"Processing video '{video_path}'...")

        # Process video frames
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Detect persons in the frame
            bboxes = self.detect_person(frame)

            # Check if any human is close
            is_close = human_is_close(img_size, bboxes, threshold=close_threshold)

            # Add frame information
            if is_close:
                frames_with_close_humans.append(frame_count)
                cv2.putText(
                    frame,
                    "HUMAN CLOSE!",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )

            # Write frame to output video
            if out:
                out.write(frame)

            # Display the frame
            if display:
                cv2.imshow("Person Detection", frame)

                # Press 'q' to exit
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_count += 1

        # Release resources
        cap.release()
        if out:
            out.release()
        if display:
            cv2.destroyAllWindows()

        print(f"Video processing completed. Processed {frame_count} frames.")
        print(f"Found {len(frames_with_close_humans)} frames with close humans.")

        return frames_with_close_humans


def human_is_close(img_size, bboxes, threshold=0.15):
    """
    Check if any human is close based on bounding box size

    Args:
        img_size: Tuple of (width, height) of the image
        bboxes: List of bounding boxes in [x1, y1, x2, y2] format
        threshold: Area threshold (0.0-1.0) for considering a human as close

    Returns:
        bool: True if any human is close, False otherwise
    """
    img_width, img_height = img_size
    img_area = img_width * img_height

    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        # Calculate area of the bounding box
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        bbox_area = bbox_width * bbox_height

        # Calculate the ratio of bbox area to image area
        area_ratio = bbox_area / img_area
        print(f"==> BBOX/IMG = {area_ratio*100:.2f}%")

        if area_ratio > threshold:
            return True

    return False


if __name__ == "__main__":
    detector = YOLODetector()
    close_human_frames = detector.process_video(
        video_path="data/video.avi",
        display=True,
        close_threshold=0.15,
    )
    exit()

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Person detection using YOLOv8")
    parser.add_argument(
        "--input", "-i", type=str, required=True, help="Path to input image"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="output/human_det.jpg",
        help="Path to output image",
    )
    parser.add_argument(
        "--display", "-d", action="store_true", help="Display output image"
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="yolov8n.pt",
        help="Path to model file or model name (yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.50,
        help="Detection confidence threshold (0.0-1.0)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run inference on ('cpu', 'cuda:0', or 'auto')",
    )
    args = parser.parse_args()

    # Check if input file exists
    if not os.path.isfile(args.input):
        print(f"Error: Input file '{args.input}' does not exist")
        exit(1)

    # Read the input image
    frame = cv2.imread(args.input)
    if frame is None:
        print(f"Error: Could not read image '{args.input}'")
        exit(1)

    # Initialize detector
    try:
        detector = YOLODetector(
            model_path=args.model, conf_threshold=args.threshold, device=args.device
        )
    except Exception as e:
        print(f"Error initializing detector: {e}")
        exit(1)

    # Time the detection
    start_time = time.time()

    # Detect people in the image
    print(f"Detecting people in '{args.input}'...")
    bboxes = detector.detect_person(frame)

    # Calculate and print elapsed time
    elapsed = time.time() - start_time
    print(f"Detection completed in {elapsed:.3f} seconds")

    # Get image size from frame
    img_height, img_width = frame.shape[:2]

    # Check if any human is close
    if human_is_close((img_width, img_height), bboxes):
        print("-->Human is close")

    # Using custom threshold (e.g., 30% of image)
    if human_is_close((img_width, img_height), bboxes, threshold=0.30):
        print("-->Human is very close")

    # Add detection info on the image
    # for i, bbox in enumerate(bboxes):
    #     x1, y1, x2, y2 = bbox
    #     # Add label with person index
    #     cv2.putText(
    #         frame,
    #         f"Person {i+1}",
    #         (x1, y1 - 10),
    #         cv2.FONT_HERSHEY_SIMPLEX,
    #         0.7,
    #         (0, 255, 0),
    #         2,
    #     )

    # Show results
    print(f"Found {len(bboxes)} people in the image")

    # Save the output image
    cv2.imwrite(args.output, frame)
    print(f"Output image saved to '{args.output}'")

    # Display if requested
    if args.display:
        cv2.imshow("Person Detection", frame)
        print("Press any key to exit...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
