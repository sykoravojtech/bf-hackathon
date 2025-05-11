import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import time

class VideoRecorder(Node):
    def __init__(self):
        super().__init__('video_recorder')
        self.subscription = self.create_subscription(
            Image,
            '/baumer/image',
            self.image_callback,
            10)
        self.bridge = CvBridge()
        self.frames = []
        self.recording_duration = 60  # seconds
        self.start_time = None
        self.get_logger().info('Video Recorder Node has been started.')

    def image_callback(self, msg):
        if self.start_time is None:
            self.start_time = time.time()

        # Convert ROS Image message to OpenCV image
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.frames.append(frame)

        # Stop recording after the specified duration
        if time.time() - self.start_time >= self.recording_duration:
            self.save_video()
            rclpy.shutdown()

    def save_video(self):
        if not self.frames:
            self.get_logger().error('No frames captured. Video not saved.')
            return

        # Get frame dimensions
        height, width, _ = self.frames[0].shape
        fps = 30  # Assuming 30 FPS for the video

        # Define the codec and create VideoWriter object
        out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

        for frame in self.frames:
            out.write(frame)

        out.release()
        self.get_logger().info('Video saved as output.avi')

def main(args=None):
    rclpy.init(args=args)
    video_recorder = VideoRecorder()
    rclpy.spin(video_recorder)
    video_recorder.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()