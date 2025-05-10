import time
from typing import Callable

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Bool, Float32, Int32, String


class Controller(Node):
    def __init__(self):
        super().__init__("robot_controller")

        # Publishers
        self.speaker_pub = self.create_publisher(String, "speaker_say", 10)
        self.rotate_pub = self.create_publisher(Float32, "rotate", 10)
        self.velocity_pub = self.create_publisher(Float32, "set_velocity", 10)
        self.display_pub = self.create_publisher(String, "show_display", 10)

        # Subscribers
        self.create_subscription(String, "odometry", self.odometry_callback, 10)
        self.create_subscription(Image, "camera", self.camera_callback, 10)
        self.create_subscription(PointCloud2, "lidar", self.lidar_callback, 10)
        self.create_subscription(String, "microphone", self.microphone_callback, 10)
        self.create_subscription(Int32, "touch", self.touch_callback, 10)

        self.robot_on = True
        self.odometry = {"position": None, "direction": None, "speed": None}
        self.camera = None
        self.lidar = None
        self.microphone = None
        self.touch = None

    def odometry_callback(self, msg):
        self.get_logger().info(f"Received odometry: {msg.data}")
        self.odometry["position"] = msg.data

    def camera_callback(self, msg):
        self.get_logger().info("Received camera data")
        self.camera = msg

    def lidar_callback(self, msg):
        self.get_logger().info("Received lidar data")
        self.lidar = msg

    def microphone_callback(self, msg):
        self.get_logger().info(f"Received microphone input: {msg.data}")
        self.microphone = msg.data

    def touch_callback(self, msg):
        self.get_logger().info(f"Received touch input: {msg.data}")
        self.touch = msg.data

    def scenario_check_loop(self):
        """
        Checks all scenarios. If one is happening run it. If none is happening just continue.
        """
        self.get_logger().info("Checking every scenario")

        try:
            user_input = input("Enter a number (or 'quit' to exit): ")
            if user_input.lower() == "quit" or user_input.lower() == "exit":
                self.robot_on = False
                self.speaker_pub.publish(String(data="Shutting down"))
                return

            try:
                number = int(user_input)
                self.get_logger().info(f"You entered: {number}")
            except ValueError:
                self.get_logger().info("Invalid input. Please enter a number or 'quit'")
                self.speaker_pub.publish(String(data="Invalid input, please try again"))
        except KeyboardInterrupt:
            self.robot_on = False
            self.speaker_pub.publish(String(data="Shutting down"))


def main(args=None):
    rclpy.init(args=args)

    controller = Controller()

    # Run the node
    try:
        while controller.robot_on:
            rclpy.spin_once(controller, timeout_sec=1)
            controller.scenario_check_loop()
    except KeyboardInterrupt:
        controller.get_logger().info("Shutting down node...")
    finally:
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
