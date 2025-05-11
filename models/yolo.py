import cv2
import numpy as np
from ultralytics import YOLO


def detect_close_human(image, model_path="yolov8n.pt", conf_threshold=0.4, close_threshold=0.21):
    model = YOLO(model_path)
    results = model(image, conf=conf_threshold, classes=0)  # Class 0 is 'person'
    img_area = image.shape[0] * image.shape[1]

    for result in results:
        for box in result.boxes:
    img_area = image.shape[0] * image.shape[1]

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            bbox_area = (x2 - x1) * (y2 - y1)
            if bbox_area / img_area > close_threshold:
                return True
    return False


class HumanDetectionNode(Node):
    def __init__(self):
        super().__init__('human_detection_node')
        self.bridge = CvBridge()
        self.stop_detection = False

        self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.create_subscription(Image, '/stop_detection', self.stop_detection_callback, 10)

    def image_callback(self, msg):
        if self.stop_detection:
            self.get_logger().info("Detection stopped.")
            return

        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        if detect_close_human(image):
            self.get_logger().info("A human is close!")
        else:
            self.get_logger().info("No close human detected.")

    def stop_detection_callback(self, msg):
        self.stop_detection = True
        self.get_logger().info("Stopping detection.")


def main(args=None):
    rclpy.init(args=args)
    node = HumanDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
