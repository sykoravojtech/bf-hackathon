import argparse
import os
import time

import cv2
import mediapipe as mp
import numpy as np

# import time
# from mediapipe.python._framework_bindings import timestamp


class real_det:
    def __init__(self, path="mobilenetv2_ssd_256_uint8.tflite", score_threshold=0.3):
        # Timestamp = timestamp.Timestamp
        model_path = path
        BaseOptions = mp.tasks.BaseOptions
        ObjectDetector = mp.tasks.vision.ObjectDetector
        ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        self.frame_index = 1
        self.fps = 30
        self.results = None
        self.score_threshold = score_threshold

        # Try using IMAGE mode first, which is more reliable for single images
        self.image_mode = True
        try:
            image_options = ObjectDetectorOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=VisionRunningMode.IMAGE,
                max_results=10,
                score_threshold=self.score_threshold,
            )
            self.image_detector = ObjectDetector.create_from_options(image_options)
        except Exception as e:
            print(f"Warning: Could not initialize image mode: {e}")
            self.image_mode = False

        # Always initialize live stream mode as fallback
        live_options = ObjectDetectorOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.LIVE_STREAM,
            max_results=10,
            score_threshold=self.score_threshold,
            result_callback=self.print_result,
        )
        self.detector = ObjectDetector.create_from_options(live_options)

    def print_result(
        self,
        result: mp.tasks.components.containers.DetectionResult,
        output_image: mp.Image,
        timestamp_ms: int,
    ):
        # print('detection result: {}'.format(result))
        self.results = result

    def detect_person(self, frame):
        bboxs = []

        # First try image mode (synchronous and more reliable)
        if self.image_mode:
            try:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                results = self.image_detector.detect(mp_image)
                if results and results.detections:
                    for res in results.detections:
                        if (
                            res.categories[0].category_name == "person"
                            and res.categories[0].score > self.score_threshold
                        ):
                            bbox = res.bounding_box
                            x1, y1, w, h = (
                                bbox.origin_x,
                                bbox.origin_y,
                                bbox.width,
                                bbox.height,
                            )
                            x2, y2 = x1 + w, y1 + h
                            bboxs.append([x1, y1, x2, y2])
                            cv2.rectangle(
                                frame, (x1, y1), (x2, y2), (0, 255, 0), thickness=2
                            )
                            score = int(res.categories[0].score * 100)
                            cv2.putText(
                                frame,
                                f"Person {score}%",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 255, 0),
                                1,
                            )
                return bboxs  # Return results from image mode if available
            except Exception as e:
                print(f"Image mode detection failed, falling back to live stream: {e}")
                # Fall through to live stream mode if image mode fails

        # Fallback to live stream mode
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        # Calculate the timestamp of the current frame
        frame_timestamp_ms = int(1000 * self.frame_index / self.fps)

        # Reset results before detection
        self.results = None

        # Perform object detection on the video frame
        self.detector.detect_async(mp_image, frame_timestamp_ms)

        # Wait a bit for results to be processed through the callback
        max_wait = 0.5  # Maximum wait time in seconds
        start_time = time.time()
        while self.results is None and time.time() - start_time < max_wait:
            time.sleep(0.01)  # Small sleep to avoid busy waiting

        if self.results:
            det_res = self.results.detections
            for res in det_res:
                if (
                    res.categories[0].category_name == "person"
                    and res.categories[0].score > self.score_threshold
                ):
                    bbox = res.bounding_box
                    x1, y1, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
                    x2, y2 = x1 + w, y1 + h
                    bboxs.append([x1, y1, x2, y2])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)
                    score = int(res.categories[0].score * 100)
                    cv2.putText(
                        frame,
                        f"Person {score}%",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )

        self.frame_index = self.frame_index + 1
        return bboxs


def human_is_close(img_size, bboxes, threshold=0.20):
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
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Person detection using MediaPipe")
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
        default="models/mobilenetv2_ssd_256_uint8.tflite",
        help="Path to model file",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.4,
        help="Detection score threshold (0.0-1.0)",
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

    # Initialize detector with the provided model path
    try:
        detector = real_det(path=args.model, score_threshold=args.threshold)
    except Exception as e:
        print(f"Error initializing detector: {e}")
        exit(1)

    # Detect people in the image
    print(f"Detecting people in '{args.input}'...")
    bboxes = detector.detect_person(frame)
    # Get image size from frame
    img_height, img_width = frame.shape[:2]

    # Using default threshold (0.20 or 20% of image)
    if human_is_close((img_width, img_height), bboxes):
        print("Human is close")

    # Using custom threshold (e.g., 30% of image)
    if human_is_close((img_width, img_height), bboxes, threshold=0.30):
        print("Human is very close")

    # Add detection info on the image
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox
        # Add label with confidence score
        cv2.putText(
            frame,
            f"Person {i+1}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

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
