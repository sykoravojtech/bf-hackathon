import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import mediapipe as mp
import numpy as np


class RealDet:
    def __init__(self, path="mobilenetv2_ssd_256_uint8.tflite", score_threshold=0.1):
        print("Initializing RealDet...")
        BaseOptions = mp.tasks.BaseOptions
        ObjectDetector = mp.tasks.vision.ObjectDetector
        ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        self.score_threshold = score_threshold
        self.image_mode = True

        try:
            image_options = ObjectDetectorOptions(
                base_options=BaseOptions(model_asset_path=path),
                running_mode=VisionRunningMode.IMAGE,
                max_results=10,
                score_threshold=self.score_threshold,
            )
            self.image_detector = ObjectDetector.create_from_options(image_options)
            print("Image detector initialized successfully.")
        except Exception as e:
            print(f"Warning: Could not initialize image mode: {e}")
            self.image_mode = False

    def detect_person(self, frame):
        print("Starting person detection...")
        bboxs = []
        if self.image_mode:
            try:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                results = self.image_detector.detect(mp_image)
                print("Detection results obtained.")
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
                            print(f"Person detected with score: {score}%")
            except Exception as e:
                print(f"Detection failed: {e}")
        else:
            print("Image mode is not enabled.")
        print("Person detection completed.")
        return bboxs


class VideoStreamProcessor(Node):
    def __init__(self):
        print("Initializing VideoStreamProcessor node...")
        super().__init__('video_stream_processor')
        self.subscription = self.create_subscription(
            Image,
            '/baumer/image',
            self.listener_callback,
            10
        )
        print("Subscription to /baumer/image topic created.")
        self.publisher = self.create_publisher(Image, '/output/image', 10)
        print("Publisher to /output/image topic created.")
        self.bridge = CvBridge()
        print("CvBridge initialized.")
        self.detector = RealDet(path="models/mobilenetv2_ssd_256_uint8.tflite", score_threshold=0.1)
        print("RealDet object created.")

    def listener_callback(self, msg):
        print("Received a new image message.")
        # Convert ROS Image message to OpenCV image
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        print("Converted ROS Image message to OpenCV image.")

        # Detect people in the frame
        bboxes = self.detector.detect_person(frame)
        print(f"Detected {len(bboxes)} person(s) in the frame.")

        # Publish the processed frame with bounding boxes
        output_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        self.publisher.publish(output_msg)
        print("Published processed frame to /output/image topic.")


def main(args=None):
    print("Starting main function...")
    rclpy.init(args=args)
    print("rclpy initialized.")
    video_stream_processor = VideoStreamProcessor()
    print("VideoStreamProcessor node created.")
    rclpy.spin(video_stream_processor)
    print("Spinning VideoStreamProcessor node...")
    video_stream_processor.destroy_node()
    print("VideoStreamProcessor node destroyed.")
    rclpy.shutdown()
    print("rclpy shutdown completed.")


if __name__ == '__main__':
    main()
