from nav2_msgs.msg import BehaviorTreeLog
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from action_msgs.msg import GoalStatusArray

class WaypointNavigator:
    def __init__(self, node):
        """
        Initialize the WaypointNavigator with an existing ROS 2 node.
        :param node: An instance of rclpy.node.Node
        """
        self.node = node

        # Publishers and subscribers
        self.goal_pub = self.node.create_publisher(PoseStamped, '/goal_pose', 10)
        self.status_sub = self.node.create_subscription(GoalStatusArray, '/move_base/status', self.status_callback, 10)
        self.confirmation_pub = self.node.create_publisher(String, '/navigation/confirmation', 10)
        self.behavior_tree_sub = self.node.create_subscription(BehaviorTreeLog, '/behavior_tree_log', self.behavior_tree_callback, 10)

        self.navigation_complete = False

        # Dictionary to store waypoints
        self.waypoints = {
            "poseA": {
            "position": {"x": 1.3124427795410156, "y": -0.18059289455413818, "z": 0.0},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.024976282819221766, "w": 0.999688043989991}
            },
            "poseB": {
            "position": {"x": 2.776034355163574, "y": 2.025285482406616, "z": 0.0},
            "orientation": {"x": 0.0, "y": 0.0, "z": -0.7526690658395774, "w": 0.65839902591679}
            },
            "base": {
            "position": {"x": 0.0, "y": 0.0, "z": 0.0},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
            }
        }

    def send_goal(self, position, orientation):
        """Send a 2D goal pose to the navigation topic."""
        print("Sending goal to navigation in sending function")
        goal = PoseStamped()
        goal.header.frame_id = "map"
        goal.header.stamp = self.node.get_clock().now().to_msg()
        goal.pose.position.x = position["x"]
        goal.pose.position.y = position["y"]
        goal.pose.position.z = position["z"]
        goal.pose.orientation.x = orientation["x"]
        goal.pose.orientation.y = orientation["y"]
        goal.pose.orientation.z = orientation["z"]
        goal.pose.orientation.w = orientation["w"]

        self.node.get_logger().info(f"Sending goal: position={position}, orientation={orientation}")
        self.goal_pub.publish(goal)
        self.navigation_complete = False

    def send_goal_by_name(self, waypoint_name):
        """Send a goal by waypoint name."""
        print("in send_goal_by_name")
        if waypoint_name in self.waypoints:
            waypoint = self.waypoints[waypoint_name]
            self.send_goal(waypoint["position"], waypoint["orientation"])
        else:
            self.node.get_logger().error(f"Waypoint '{waypoint_name}' not found.")

    def update_waypoint(self, waypoint_name, position, orientation):
        """Update the position and orientation of a waypoint."""
        if waypoint_name in self.waypoints:
            self.waypoints[waypoint_name]["position"] = position
            self.waypoints[waypoint_name]["orientation"] = orientation
            self.node.get_logger().info(f"Updated waypoint '{waypoint_name}' to position={position}, orientation={orientation}")
        else:
            self.node.get_logger().error(f"Waypoint '{waypoint_name}' not found.")

    def status_callback(self, msg):
        """Callback to check the status of the navigation."""
        for status in msg.status_list:
            if status.status == 3:  # Status 3 means the goal was reached
                self.node.get_logger().info("Navigation goal reached.")
                self.navigation_complete = True
                self.publish_confirmation()
                break

    def behavior_tree_callback(self, msg):
        """Callback to process BehaviorTreeLog messages."""
        for entry in msg.behavior_tree_status_changes:
            if entry.status == "SUCCESS":  # Check for successful navigation
                self.node.get_logger().info("Behavior tree indicates navigation success.")
                self.navigation_complete = True
                self.publish_confirmation()
                break

    def publish_confirmation(self):
        """Publish confirmation that navigation is complete."""
        confirmation_msg = String()
        confirmation_msg.data = "Navigation complete"
        self.confirmation_pub.publish(confirmation_msg)
        self.node.get_logger().info("Published navigation confirmation.")
