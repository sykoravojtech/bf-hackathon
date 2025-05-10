#!/usr/bin/env python3

import rclpy
from action_msgs.msg import GoalStatusArray
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from std_msgs.msg import String


class WaypointNavigator(Node):
    def __init__(self):
        super().__init__("waypoint_navigator")

        # Publishers and subscribers
        self.goal_pub = self.create_publisher(PoseStamped, "/move_base_simple/goal", 10)
        self.status_sub = self.create_subscription(
            GoalStatusArray, "/move_base/status", self.status_callback, 10
        )
        self.confirmation_pub = self.create_publisher(
            String, "/navigation/confirmation", 10
        )

        self.navigation_complete = False

        # Dictionary to store waypoints
        self.waypoints = {
            "poseA": (1.0, 2.0, 0.0),
            "poseB": (3.0, 4.0, 0.0),
            "base": (0.0, 0.0, 0.0),
        }

    def send_goal(self, x, y, theta):
        """Send a 2D goal pose to the navigation topic."""
        goal = PoseStamped()
        goal.header.frame_id = "map"
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.orientation.z = (
            theta  # Assuming theta is in quaternion form or needs conversion
        )
        goal.pose.orientation.w = 1.0  # Assuming no rotation for simplicity

        self.get_logger().info(f"Sending goal: x={x}, y={y}, theta={theta}")
        self.goal_pub.publish(goal)
        self.navigation_complete = False

    def send_goal_by_name(self, waypoint_name):
        """Send a goal by waypoint name."""
        if waypoint_name in self.waypoints:
            x, y, theta = self.waypoints[waypoint_name]
            self.send_goal(x, y, theta)
        else:
            self.get_logger().error(f"Waypoint '{waypoint_name}' not found.")

    def status_callback(self, msg):
        """Callback to check the status of the navigation."""
        for status in msg.status_list:
            if status.status == 3:  # Status 3 means the goal was reached
                self.get_logger().info("Navigation goal reached.")
                self.navigation_complete = True
                self.publish_confirmation()
                break

    def publish_confirmation(self):
        """Publish confirmation that navigation is complete."""
        confirmation_msg = String()
        confirmation_msg.data = "Navigation complete"
        self.confirmation_pub.publish(confirmation_msg)
        self.get_logger().info("Published navigation confirmation.")


def main(args=None):
    rclpy.init(args=args)
    navigator = WaypointNavigator()

    # Example usage
    try:
        navigator.send_goal_by_name("poseA")  # Send goal using waypoint name
        rclpy.spin(navigator)
    except KeyboardInterrupt:
        navigator.get_logger().info("Waypoint Navigator node terminated.")
    finally:
        navigator.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
