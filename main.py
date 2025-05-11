import time
from typing import Callable
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Int32, Float32, Bool
from sensor_msgs.msg import PointCloud2, Image
from nav2_msgs.msg import BehaviorTreeLog
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from action_msgs.msg import GoalStatusArray
from cv_bridge import CvBridge
from gtts import gTTS
from playsound import playsound
import pyttsx3
import speech_recognition as sr
from vosk import Model, KaldiRecognizer
import sounddevice as sd
import queue
import json
import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
import numpy as np
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage
import time
from pathlib import Path

# Import waypoint navigation functionality
# from functions.waypoint import WaypointNavigator  # Assuming functions/waypoint.py contains this class

class Controller(Node):
    def __init__(self):
        super().__init__('controller')  # Initialize the Node with a name
        print("Initializing Controller...")
        # Publishers
        print("Creating publishers...")
        self.speaker_pub = self.create_publisher(String, "/speaker_say", 10)
        self.rotate_pub = self.create_publisher(Float32, "/rotate", 10)
        self.velocity_pub = self.create_publisher(Float32, "/set_velocity", 10)
        self.display_pub = self.create_publisher(String, "/show_display", 10)
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.confirmation_pub = self.create_publisher(String, '/navigation/confirmation', 10)
        self.image_pub_human = self.create_publisher(Image, "/output_image_human", 10)
        self.image_pub_hands = self.create_publisher(Image, "/output_image_hands", 10)

        # Subscribers
        print("Creating subscribers...")
        self.create_subscription(String, "odometry", self.odometry_callback, 10)
        self.create_subscription(Image, "camera", self.camera_callback, 10)
        self.create_subscription(PointCloud2, "lidar", self.lidar_callback, 10)
        self.create_subscription(String, "microphone", self.microphone_callback, 10)
        self.create_subscription(Int32, "touch", self.touch_callback, 10)
        self.create_subscription(BehaviorTreeLog, '/behavior_tree_log', self.behavior_tree_callback, 10)
        self.create_subscription(Image, '/baumer/camera', self.image_callback_human, 10)
        self.create_subscription(Image, "/baumer/camera", self.image_callback_hands, 10
        self.create_subscription(Image, '/stop_detection_human', self.stop_detection_callback, 10)

        self.robot_on = True
        self.odometry = {"position": None, "direction": None, "speed": None}
        self.behavior_tree_msg = None  # Initialize behavior_tree_msg
        self.camera = None
        self.lidar = None
        self.microphone = None
        self.touch = None
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

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(
        static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5
        )
        self.detection_enabled_hands = False



        # Load YOLOv8 model
        model_path="/home/zany/workspace/src/bf-hackathon/models/vosk-model-small-en-us-0.15"
        self.model = YOLO(model_path)

        self.bridge = CvBridge()
        self.stop_detection_human = False

        # Initialize waypoint navigator
        print("Initializing waypoint navigator...")

    def odometry_callback(self, msg):
        print("Odometry callback triggered.")
        self.get_logger().info(f"Received odometry: {msg.data}")
        self.odometry["position"] = msg.data

    def camera_callback(self, msg):
        print("Camera callback triggered.")
        self.get_logger().info("Received camera data")
        self.camera = msg

    def lidar_callback(self, msg):
        print("Lidar callback triggered.")
        self.get_logger().info("Received lidar data")
        self.lidar = msg

    def microphone_callback(self, msg):
        print("Microphone callback triggered.")
        self.get_logger().info(f"Received microphone input: {msg.data}")
        self.microphone = msg.data

    def touch_callback(self, msg):
        print("Touch callback triggered.")
        self.get_logger().info(f"Received touch input: {msg.data}")
        self.touch = msg.data

    def behavior_tree_callback(self, msg):
        """
        Process behavior tree log messages and publish navigation status updates.
        This function will only exit when a final outcome is received.
        """
        self.behavior_tree_msg = msg
        print("Behavior tree callback triggered.")

    def send_goal(self, position, orientation):
        """Send a 2D goal pose to the navigation topic."""
        print("Entering send_goal")
        goal = PoseStamped()
        goal.header.frame_id = "map"
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = position["x"]
        goal.pose.position.y = position["y"]
        goal.pose.position.z = position["z"]
        goal.pose.orientation.x = orientation["x"]
        goal.pose.orientation.y = orientation["y"]
        goal.pose.orientation.z = orientation["z"]
        goal.pose.orientation.w = orientation["w"]

        self.get_logger().info(f"Sending goal: position={position}, orientation={orientation}")
        self.goal_pub.publish(goal)
        self.navigation_complete = False

    def send_goal_by_name(self, waypoint_name):
        """Send a goal by waypoint name."""
        print("Entering send_goal_by_name")
        if waypoint_name in self.waypoints:
            waypoint = self.waypoints[waypoint_name]
            print(f"Found waypoint: {waypoint_name}")
            self.send_goal(waypoint["position"], waypoint["orientation"])
        else:
            print(f"Waypoint '{waypoint_name}' not found")
            self.get_logger().error(f"Waypoint '{waypoint_name}' not found.")
        print("Exiting send_goal_by_name")
    
    def text_to_speech(self, text, lang='en'):
        tts = gTTS(text=text, lang=lang)
        tts.save("output.mp3")
        playsound("output.mp3")  # Play the audio immediately

    def robot_pipeline(self, input_mode="speech", offline=False, model_path="/home/zany/workspace/src/bf-hackathon/models/vosk-model-small-en-us-0.15"):
        # Define commands
        commands = {
            "turn left": "robot is turning left",
            "turn right": "robot is turning right",
            "stop": "robot has stopped",
            "wait": "robot is waiting for a moment",
            "move forward": "robot is moving forward",
            "move backward": "robot is moving backward",
            "start": "robot is starting",
            "pause": "robot is paused",
            "go to sleep": "robot is sleeping",
            "wake up": "robot is waking up",
            "bring me beer": "beer coming up",
            "bring me coffee": "coffee coming up",
            "move right": "robot is moving to the right",
            "move left": "robot is moving to the left",
            "can you dance": "Of course, if you show me how to dance"
        }

        # TTS engine
        tts_engine = pyttsx3.init()

        def respond(text):
            print(f"🤖 {text}")
            tts_engine.say(text)
            tts_engine.runAndWait()

        def process_command(text):
            for keyword, response in commands.items():
                if keyword in text:
                    return response
            return "Sorry, I did not understand the command."

        def listen_command_online():
            recognizer = sr.Recognizer()
            with sr.Microphone() as source:
                print("🎤 Listening (online)...")
                audio = recognizer.listen(source)
            try:
                return recognizer.recognize_google(audio).lower()
            except:
                return "Speech recognition failed (online)."

        def listen_command_offline_vosk():
            q = queue.Queue()
            model = Model(model_path)
            rec = KaldiRecognizer(model, 16000)

            def callback(indata, frames, time, status):
                if status:
                    print(status)
                q.put(bytes(indata))

            print("🎤 Listening (offline)...")
            with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                                channels=1, callback=callback):
                while True:
                    data = q.get()
                    if rec.AcceptWaveform(data):
                        result = json.loads(rec.Result())
                        return result.get("text", "").lower()

        if input_mode == "text":
            user_input = input("⌨️ Type your command: ").lower()
            print(f"🗣️ You typed: {user_input}")  # Show typed input
            response = f"Got it, you said: {user_input}"  # The robot repeats back the input
        elif input_mode == "speech":
            if offline:
                user_input = listen_command_offline_vosk()
            else:
                user_input = listen_command_online()
            print(f"🗣️ You said: {user_input}")
            response = process_command(user_input)  # Process speech command to a response
        else:
            respond("Invalid input mode.")
            return

        print(f"🤖 Response: {response}")  # Print the response
        respond(response)  # Speak the response
    
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

    def image_callback_hands(self, msg: Image):
        # Store the latest image for processing
        self.latest_image = msg

        if self.detection_enabled_hands:
            self.process_image(msg)

    def process_image(self, msg: Image):
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



    def scenario_check_loop(self):
        """
        Checks all scenarios. If one is happening run it. If none is happening just continue.
        """
        print("Entering scenario_check_loop...")

        self.get_logger().info("Checking every scenario")

        try:
            user_input = input("Enter a number (or 'quit' to exit): ")
            print(f"User input: {user_input}")
            if user_input.lower() == "quit":
                print("User chose to quit.")
                self.robot_on = False
                self.speaker_pub.publish(String(data="Shutting down"))
                return

            try:
                number = int(user_input)
                print(f"Parsed number: {number}")
                self.get_logger().info(f"You entered: {number}")
            except ValueError:
                print("Invalid input detected.")
                self.get_logger().info("Invalid input. Please enter a number or 'quit'")
                self.speaker_pub.publish(String(data="Invalid input, please try again"))
        except KeyboardInterrupt:
            print("KeyboardInterrupt detected in scenario_check_loop.")
            self.robot_on = False
            self.speaker_pub.publish(String(data="Shutting down"))

    def navigate_to_waypoints(self):
        """
        Navigate to predefined waypoints A and B with confirmation.
        """
        print("Starting navigation to waypoints...")
        self.get_logger().info("Navigating to waypoint A...")
        self.send_goal_by_name("serve_location")  # Replace "A" with actual coordinates if needed

        if self.behavior_tree_msg is None:
            self.get_logger().error("Behavior tree message is not received yet.")
            return

        for event in self.behavior_tree_msg.event_log:
            if (
            event.node_name == "NavigateRecovery" and
            event.previous_status == "SUCCESS" and
            event.current_status == "IDLE"
            ):
                self.get_logger().info("Navigation to waypoint A completed successfully.")
                self.navigation_complete = True
                break
            if (
            event.node_name == "NavigateRecovery" and
            event.previous_status == "FAILURE" and
            event.current_status == "IDLE"
            ):
                self.get_logger().info("Navigation to waypoint A FAILED.")
                self.navigation_complete = False
                break
            self.get_logger().info("Navigation Exited.")
            break

        # waypoint_navigator.navigation_complete = True

        if self.navigation_complete:
            self.get_logger().info("Navigating to waypoint B...")
            self.send_goal_by_name("fridge1")  # Replace "B" with actual coordinates if needed
            if self.navigation_complete:
                self.get_logger().info("Navigation to waypoints completed successfully.")
            else:
                self.get_logger().error("Failed to navigate to waypoint B.")
        else:
            self.get_logger().error("Failed to navigate to waypoint A.")
        
    def detect_close_human(self, image, model_path="yolov8n.pt", conf_threshold=0.4, close_threshold=0.21):
        model = YOLO(model_path)
        results = model(image, conf=conf_threshold, classes=0)  # Class 0 is 'person'
        img_area = image.shape[0] * image.shape[1]

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                bbox_area = (x2 - x1) * (y2 - y1)
                if bbox_area / img_area > close_threshold:
                    return True
        return False
    
    def image_callback_human(self, msg):
        if self.stop_detection_human:
            self.get_logger().info("Detection stopped.")
            return
        self.image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        if self.detect_close_human(self.image):
            self.human_detected = True
            self.get_logger().info("A human is close!")
        else:
            self.get_logger().info("No close human detected.")

    # def stop_detection_callback(self, msg):
        # self.stop_detection = True
        # self.get_logger().info("Stopping detection.")
    
    def image_callback_hands(self, msg: Image):
        # Store the latest image for processing
        self.latest_image = msg

        if self.detection_enabled:
            self.process_image(msg)
        

def main(args=None):
    print("Starting main function...")
    rclpy.init(args=args)

    print("Creating Controller instance...")
    controller = Controller()

    # Run the node
    try:
        while controller.robot_on:
            rclpy.spin_once(controller, timeout_sec=1)
            # controller.send_goal_by_name("base_pose")
            controller.send_goal_by_name("serve_location")

            controller.text_to_speech("Hello! I am Sewi, your interactive robot assistant. Would you like to have a Bier or coffee?", lang='en')
            # controller.navigate_to_waypoints()
            # self.stop_detection_human = False
            # if human_detected:
            controller.robot_pipeline(input_mode="speech", offline=False, model_path="/home/zany/workspace/sew_ws/src/bf-hackathon/models/vosk-model-small-en-us-0.15")
            controller.text_to_speech("Please confirm me that you want a coffee or a beer by showing me a thumbs up or clicking on the screen", lang='en')
            # controller.get_logger().info("Starting human detection...")
            controller.send_goal_by_name("fridge1")
            controller.text_to_speech("Hello! Can you please give me a cup of coffee", lang='en')
            controller.text_to_speech("Can you also please confirm me that you have put the coffee on my surface, you can either show me a thumbs up or click on the screen", lang='en')
            controller.text_to_speech("Thank you for the coffee, I am going to serve it now", lang='en')
            controller.send_goal_by_name("serve_location")
            controller.text_to_speech("Please take the coffee", lang='en')
            controller.text_to_speech("If you liked it, show me a victory", lang='en')
            controller.text_to_speech("Thank You, Have a nice day!", lang='en')
    
            
            
            
            controller.scenario_check_loop()
    except KeyboardInterrupt:
        controller.get_logger().info("Shutting down node...")
    finally:
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    print("Executing script...")
    main()
