import cv2
import numpy as np


def save_frame(frame, output_path):
    """
    Save a single frame to disk.

    Args:
        frame (numpy.ndarray): Image to save
        output_path (str): Path where the image will be saved

    Returns:
        bool: True if successful, False otherwise
    """
    if frame is None:
        return False

    try:
        cv2.imwrite(output_path, frame)
        return True
    except Exception as e:
        print(f"Error saving frame: {e}")
        return False


def process_frame(frame, resize=None, grayscale=False):
    """
    Process a video frame with basic operations.

    Args:
        frame (numpy.ndarray): Input frame
        resize (tuple, optional): Target size (width, height)
        grayscale (bool): Whether to convert to grayscale

    Returns:
        numpy.ndarray: Processed frame
    """
    if frame is None:
        return None

    result = frame.copy()

    # Convert to grayscale if requested
    if grayscale:
        result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    # Resize if dimensions provided
    if resize is not None:
        result = cv2.resize(result, resize)

    return result


def get_frame_from_video(video_source, frame_number=0):
    """
    Extract a specific frame from a video file or camera.

    Args:
        video_source: Path to video file or camera index
        frame_number: Frame number to extract (0 for first frame)

    Returns:
        numpy.ndarray: Extracted frame or None if failed
    """
    try:
        cap = cv2.VideoCapture(video_source)

        if not cap.isOpened():
            print(f"Error: Could not open video source {video_source}")
            return None

        # Set position to the specified frame
        if frame_number > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        # Read the frame
        ret, frame = cap.read()
        cap.release()

        if ret:
            return frame
        else:
            print(f"Error: Could not read frame {frame_number} from {video_source}")
            return None

    except Exception as e:
        print(f"Error extracting frame: {e}")
        return None


class VideoReader:
    """
    Video reader class for processing video files or camera streams.
    This provides efficient access to video frames.
    """

    def __init__(self, video_source):
        """
        Initialize the video reader with a source.

        Args:
            video_source: Path to video file or camera index (0 for default camera)
        """
        self.video_source = video_source
        self.cap = cv2.VideoCapture(video_source)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video source: {video_source}")

        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # For webcams, frame_count might be 0
        if self.frame_count == 0:
            self.frame_count = float("inf")

    def __del__(self):
        """Release resources upon destruction"""
        if hasattr(self, "cap") and self.cap.isOpened():
            self.cap.release()

    def read_frame(self):
        """
        Read the next frame from the video source.

        Returns:
            numpy.ndarray or None: Frame image if successful, None otherwise
        """
        if not self.cap.isOpened():
            return None

        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def read_frames(self, count=1):
        """
        Read multiple consecutive frames.

        Args:
            count (int): Number of frames to read

        Returns:
            list: List of frames (will be smaller if fewer frames available)
        """
        frames = []
        for _ in range(count):
            frame = self.read_frame()
            if frame is None:
                break
            frames.append(frame)
        return frames

    def set_frame_position(self, frame_number):
        """
        Set the position to a specific frame in the video.

        Args:
            frame_number (int): Frame number to seek to

        Returns:
            bool: True if successful, False otherwise
        """
        if not self.cap.isOpened() or frame_number >= self.frame_count:
            return False
        return self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    def release(self):
        """Release the video capture resources."""
        if self.cap.isOpened():
            self.cap.release()


# ============ JUST TESTING =========
if __name__ == "__main__":
    # Path to the warehouse video
    warehouse_video = "data/warehouse/warehouse_vid.mp4"

    # Example 1: Extract a single frame from the warehouse video
    print("Example 1: Extracting a single frame")
    frame = get_frame_from_video(warehouse_video)
    if frame is not None:
        print("- Successfully extracted the first frame")
        # Only save if explicitly requested
        # save_frame(frame, "output/warehouse_frame_0.jpg")
    else:
        print("- Failed to extract frame")

    # Example 2: Extract a specific frame from the video
    print("\nExample 2: Extracting a specific frame")
    frame_100 = get_frame_from_video(warehouse_video, frame_number=100)
    if frame_100 is not None:
        # Process the frame - resize to 640x480 and convert to grayscale
        processed_frame = process_frame(frame_100, resize=(640, 480), grayscale=True)
        print("- Successfully extracted and processed frame 100")
    else:
        print("- Failed to extract frame 100")

    # Example 3: Using VideoReader to extract multiple frames
    print("\nExample 3: Extracting multiple frames")
    try:
        video = VideoReader(warehouse_video)

        # Print video information
        print(f"- Video properties:")
        print(f"  - Frame count: {video.frame_count}")
        print(f"  - FPS: {video.fps}")
        print(f"  - Resolution: {video.width}x{video.height}")

        # Extract frames at specific intervals
        frames = []
        for i in range(
            0, min(video.frame_count, 500), 100
        ):  # Get every 100th frame up to 500
            video.set_frame_position(i)
            frame = video.read_frame()
            if frame is not None:
                frames.append(frame)

        print(f"- Extracted {len(frames)} frames")
        video.release()
    except Exception as e:
        print(f"- Error in Example 3: {e}")

    # Example 4: Process consecutive frames using VideoReader
    print("\nExample 4: Processing consecutive frames")
    try:
        video = VideoReader(warehouse_video)

        # Get 5 consecutive frames
        frames = video.read_frames(5)

        # Process each frame (no text annotations)
        processed_frames = []
        for frame in frames:
            # Process frame without adding any text
            processed = process_frame(frame, resize=(320, 240))
            processed_frames.append(processed)

        print(f"- Processed {len(processed_frames)} consecutive frames")
        video.release()
    except Exception as e:
        print(f"- Error in Example 4: {e}")

    # Example 5: Comparing the first and last frames
    print("\nExample 5: Extracting first and last frames")
    try:
        video = VideoReader(warehouse_video)

        # Get the first frame
        first_frame = video.read_frame()

        # Go to the last frame
        video.set_frame_position(video.frame_count - 1)
        last_frame = video.read_frame()

        if first_frame is not None and last_frame is not None:
            # Process frames if needed
            first_frame_resized = process_frame(first_frame, resize=(320, 240))
            last_frame_resized = process_frame(last_frame, resize=(320, 240))
            print("- Successfully extracted first and last frames")

        video.release()
    except Exception as e:
        print(f"- Error in Example 5: {e}")

    print("\nAll examples completed. Frames were extracted without saving to disk.")
