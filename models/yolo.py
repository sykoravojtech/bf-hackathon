import argparse
import os
import time

import cv2
import numpy as np

# Add ROS imports
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
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

        # Initialize ROS node, publisher and cv_bridge
        if not rospy.core.is_initialized():
            rospy.init_node("yolo_detector", anonymous=True)
        self.bridge = CvBridge()
        self.image_pub = rospy.Publisher("human_detection/image", Image, queue_size=10)
        self.image_sub = rospy.Subscriber(
            "camera/image_raw", Image, self.image_callback
        )
        rospy.loginfo("YOLO detector initialized and subscribed to camera/image_raw")

    def image_callback(self, data):
        """
        Process incoming ROS image messages

        Args:
            data: ROS Image message
        """
        try:
            # Convert ROS Image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return

        # Process image with YOLO detector
        processed_image = cv_image.copy()
        bboxes = self.detect_person(processed_image)

        # Check if any human is close
        img_height, img_width = processed_image.shape[:2]
        is_close = human_is_close((img_width, img_height), bboxes, threshold=0.21)

        # Add warning text if human is close
        if is_close:
            cv2.putText(
                processed_image,
                "HUMAN CLOSE!",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
        else:
            cv2.putText(
                processed_image,
                ":)",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

        try:
            # Convert processed image back to ROS message and publish
            ros_image = self.bridge.cv2_to_imgmsg(processed_image, "bgr8")
            ros_image.header = data.header  # Maintain original header
            self.image_pub.publish(ros_image)
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")

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
    detector = YOLODetector()
    try:
        # Spin and process callbacks until shutdown
        rospy.loginfo("YOLO detector running. Waiting for images...")
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down YOLO detector")
