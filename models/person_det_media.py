import argparse
import os

import cv2
import mediapipe as mp
import numpy as np

# import time
# from mediapipe.python._framework_bindings import timestamp


class real_det:
    def __init__(self, path="mobilenetv2_ssd_256_uint8.tflite"):
        # Timestamp = timestamp.Timestamp
        model_path = path
        BaseOptions = mp.tasks.BaseOptions
        ObjectDetector = mp.tasks.vision.ObjectDetector
        ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        self.frame_index = 1
        self.fps = 30
        self.results = None

        options = ObjectDetectorOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.LIVE_STREAM,
            max_results=5,
            result_callback=self.print_result,
        )

        self.detector = ObjectDetector.create_from_options(options)

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
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        # Calculate the timestamp of the current frame
        frame_timestamp_ms = int(1000 * self.frame_index / self.fps)
        # print(ts.seconds())
        # Perform object detection on the video frame.
        self.detector.detect_async(mp_image, frame_timestamp_ms)
        if self.results != None:
            # print('detection result1: {}'.format(self.results))
            det_res = self.results.detections
            for res in det_res:
                # print(res)
                if (
                    res.categories[0].category_name == "person"
                    and res.categories[0].score > 0.6
                ):
                    bbox = res.bounding_box
                    x1, y1, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
                    x2, y2 = x1 + w, y1 + h
                    bboxs.append([x1, y1, x2, y2])
                    cv2.rectangle(
                        frame, (x1, y1), (x2, y2), (255, 255, 255), thickness=2
                    )
                    # cv2.putText(image, str(bbox[1]), (bbox[0][0], int(bbox[0][1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                    # print("------------")
        self.frame_index = self.frame_index + 1
        return bboxs


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
        default="output_detection.jpg",
        help="Path to output image",
    )
    parser.add_argument(
        "--display", "-d", action="store_true", help="Display output image"
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="models/mediapipe_old/hackathon_files/mobilenetv2_ssd_256_uint8.tflite",
        help="Path to model file",
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
        detector = real_det(path=args.model)
    except Exception as e:
        print(f"Error initializing detector: {e}")
        exit(1)

    # Detect people in the image
    print(f"Detecting people in '{args.input}'...")
    bboxes = detector.detect_person(frame)

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
