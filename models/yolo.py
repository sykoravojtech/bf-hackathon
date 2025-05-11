import cv2
import numpy as np
from ultralytics import YOLO
from rclpy.node import Node
import rclpy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy, QoSReliabilityPolicy
# This is a ROS2 node that detects humans in images using YOLOv8 and stops detection based on a message.


def detect_close_human(image, model_path="yolov8n.pt", conf_threshold=0.4, close_threshold=0.21):
    print("detect_close_human: Loading YOLO model...")
    model = YOLO(model_path)
    print("detect_close_human: Running model inference...")
    results = model(image, conf=conf_threshold, classes=0)  # Class 0 is 'person'
    img_area = image.shape[0] * image.shape[1]
    print(f"detect_close_human: Image area = {img_area}")

    for result in results:
        print("detect_close_human: Processing result...")
        for box in result.boxes:
            print(f"detect_close_human: Found box with coordinates {box.xyxy[0].tolist()}")
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            bbox_area = (x2 - x1) * (y2 - y1)
            print(f"detect_close_human: Bounding box area = {bbox_area}")
            if bbox_area / img_area > close_threshold:
                print("detect_close_human: Human detected close!")
                return True
    print("detect_close_human: No close human detected.")
    return False


class HumanDetectionNode(Node):
    def __init__(self):
        print("HumanDetectionNode: Initializing node...")
        super().__init__('human_detection_node')
        self.bridge = CvBridge()
        self.stop_detection = False

        print("HumanDetectionNode: Creating subscriptions...")
        self.create_subscription(Image, '/baumer/image', self.image_callback, 10)
        self.create_subscription(Image, '/stop_detection', self.stop_detection_callback, 10)

    def image_callback(self, msg):
        print("image_callback: Received image message.")
        if self.stop_detection:
            print("image_callback: Detection is stopped.")
            self.get_logger().info("Detection stopped.")
            return

        print("image_callback: Converting ROS image to OpenCV format...")
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        print("image_callback: Running human detection...")
        if detect_close_human(image):
            print("image_callback: Human detected close!")
            self.get_logger().info("A human is close!")
        else:
            print("image_callback: No close human detected.")
            self.get_logger().info("No close human detected.")

    def stop_detection_callback(self, msg):
        print("stop_detection_callback: Received stop detection message.")
        self.stop_detection = True
        self.get_logger().info("Stopping detection.")

def main(args=None):
    print("main: Initializing ROS2...")
    rclpy.init(args=args)
    print("main: Creating HumanDetectionNode...")
    node = HumanDetectionNode()
    print("main: Spinning node...")
    rclpy.spin(node)
    print("main: Destroying node...")
    node.destroy_node()
    print("main: Shutting down ROS2...")
    rclpy.shutdown()


if __name__ == '__main__':
    print("Starting script...")
    main()
