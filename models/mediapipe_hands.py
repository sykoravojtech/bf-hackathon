#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import mediapipe as mp
import numpy as np

class HandGestureNode(Node):
    def __init__(self):
        super().__init__("hand_gesture_node")
        self.bridge = CvBridge()

        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(
            static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5
        )

        # ROS 2 Subscribers and Publishers
        self.image_sub = self.create_subscription(
            Image, "/baumer/image", self.image_callback, 10
        )
        self.image_pub = self.create_publisher(Image, "/output_image_topic", 10)

    def recognize_gesture(self, landmarks) -> str:
        # Gesture recognition logic (same as in the given program)
        wrist = landmarks.landmark[0]
        thumb_tip = landmarks.landmark[4]
        index_tip = landmarks.landmark[8]
        middle_tip = landmarks.landmark[12]
        ring_tip = landmarks.landmark[16]
        pinky_tip = landmarks.landmark[20]

        thumb_extended = (
            thumb_tip.x < wrist.x
            if landmarks.landmark[17].x < wrist.x
            else thumb_tip.x > wrist.x
        )
        index_extended = index_tip.y < landmarks.landmark[5].y
        middle_extended = middle_tip.y < landmarks.landmark[9].y
        ring_extended = ring_tip.y < landmarks.landmark[13].y
        pinky_extended = pinky_tip.y < landmarks.landmark[17].y

        if (
            not thumb_extended
            and index_extended
            and middle_extended
            and ring_extended
            and pinky_extended
        ):
            return "Palm"
        elif (
            thumb_tip.y < landmarks.landmark[2].y
            and all(
                abs(landmarks.landmark[i].x - landmarks.landmark[i - 3].x) < 0.05
                for i in [8, 12, 16, 20]
            )
        ):
            return "Thumbs Up"
        elif (
            not thumb_extended
            and index_extended
            and not middle_extended
            and not ring_extended
            and not pinky_extended
        ):
            return "Pointing"
        elif (
            thumb_extended
            and index_extended
            and middle_extended
            and not ring_extended
            and not pinky_extended
        ):
            return "Peace"
        elif (
            not thumb_extended
            and not index_extended
            and not middle_extended
            and not ring_extended
            and not pinky_extended
        ):
            return "Fist"
        else:
            return "Unknown Gesture"

    def image_callback(self, msg: Image):
        # Convert ROS Image message to OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        # Process the image with MediaPipe Hands
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_image)

        annotated_image = cv_image.copy()
        gesture_text = "No hands detected"

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                # Draw landmarks
                self.mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style(),
                )

                # Recognize gesture
                gesture_text = self.recognize_gesture(hand_landmarks)
                self.get_logger().info(f"Recognized gesture: {gesture_text}")

        # Overlay gesture text on the image
        cv2.putText(
            annotated_image,
            gesture_text,
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

        # Publish the annotated image
        output_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding="bgr8")
        self.image_pub.publish(output_msg)


def main(args=None):
    rclpy.init(args=args)
    node = HandGestureNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
