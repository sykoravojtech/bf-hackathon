from nav2_msgs.msg import BehaviorTreeLog
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from action_msgs.msg import GoalStatusArray
import rclpy

class WaypointNavigator:
    def __init__(self, node):
        """
        Initialize the WaypointNavigator with an existing ROS 2 node.
        :param node: An instance of rclpy.node.Node
        """
        print("Initializing WaypointNavigator")
        self.node = node

        # Publishers and subscribers
        self.goal_pub = self.node.create_publisher(PoseStamped, '/goal_pose', 10)
        self.confirmation_pub = self.node.create_publisher(String, '/navigation/confirmation', 10)
        self.behavior_tree_sub = self.node.create_subscription(BehaviorTreeLog, '/behavior_tree_log', self.behavior_tree_callback, 10)

        self.waypoints = {
            "base_pose": {
            "position": {"x": 0.0, "y": 0.0, "z": 0.0},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
            },
            "serve_location": {
            "position": {"x": 0.8551623821258545, "y": -0.455843448638916, "z": 0.0},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.04786416406388591, "w": 0.9988538540739909}
            },
            "fridge1": {
            "position": {"x": 2.897230625152588, "y": 2.248413562774658, "z": 0.0},
            "orientation": {"x": 0.0, "y": 0.0, "z": -0.8103592937024221, "w": 0.585933285545472}
            }
        }

    def behavior_tree_callback(self, msg):
        """
        Process behavior tree log messages and publish navigation status updates.
        This function will only exit when a final outcome is received.
        """
        self.behavior_tree_msg = msg
        print("Behavior tree callback triggered.")
        # final_outcome_received = False

        # previous_status = None
        # while not final_outcome_received:
        #     for event in msg.event_log:
        #     if event.node_name == "NavigateRecovery":
        #         if previous_status in ["SUCCESS", "FAILURE"] and event.current_status == "IDLE":
        #         final_outcome_received = True
        #         self.node.get_logger().info(f"Final outcome: {previous_status} -> {event.current_status}")
        #         self.confirmation_pub.publish(String(data=f"Navigation {previous_status.lower()} -> {event.current_status.lower()}"))
        #         break
        #         elif event.current_status in ["RUNNING", "PENDING", "UNKNOWN"]:
        #         self.node.get_logger().info(f"Current status: {event.current_status}")
        #         self.confirmation_pub.publish(String(data=f"Navigation {event.current_status.lower()}"))
        #         previous_status = event.current_status
    
    def send_goal(self, position, orientation):
        """Send a 2D goal pose to the navigation topic."""
        print("Entering send_goal")
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
       

        # timeout = 30.0  # Timeout in seconds
        # start_time = self.node.get_clock().now().nanoseconds / 1e9

        # while not self.navigation_complete:
        #     # rclpy.spin_once(self.node, timeout_sec=1.0)
        #     # current_time = self.node.get_clock().now().nanoseconds / 1e9
        #     # if current_time - start_time > timeout:
        #         # self.node.get_logger().error("Navigation timed out.")
        #     if self.navigation_complete:
        #         self.node.get_logger().info("Navigation succeeded.")
        #         return True  # Indicate success
        #     else:
        #         self.node.get_logger().error("Navigation failed.")
        #         return False  # Indicate failure
        print("Exiting send_goal")

    def send_goal_by_name(self, waypoint_name):
        """Send a goal by waypoint name."""
        print("Entering send_goal_by_name")
        if waypoint_name in self.waypoints:
            waypoint = self.waypoints[waypoint_name]
            print(f"Found waypoint: {waypoint_name}")
            self.send_goal(waypoint["position"], waypoint["orientation"])
        else:
            print(f"Waypoint '{waypoint_name}' not found")
            self.node.get_logger().error(f"Waypoint '{waypoint_name}' not found.")
        print("Exiting send_goal_by_name")


