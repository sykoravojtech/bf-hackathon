#!/usr/bin/env python3
"""
Process images with MediaPipe Hands to detect hand landmarks.

MediaPipe Hands detects 21 3D landmarks on each hand:
- Wrist (1 point)
- Thumb (4 points)
- Index finger (4 points)
- Middle finger (4 points)
- Ring finger (4 points)
- Pinky finger (4 points)

These landmarks can be used to recognize various hand gestures through post-processing
analysis of the relative positions of the landmarks.
"""
import argparse
import os
import sys
from typing import Any, Dict, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


def process_image(
    image_path: str,
    save_output: bool = False,
    output_path: Optional[str] = None,
    display_image: bool = False,
) -> Dict[str, Any]:
    """
    Process an image with MediaPipe Hands and show the 21 detected hand landmarks.

    Args:
        image_path: Path to the input image file
        save_output: Whether to save the output image
        output_path: Path to save the annotated image (if save_output is True)
        display_image: Whether to display the output image using cv2.imshow

    Returns:
        Dictionary containing hand landmarks and detection results
    """
    # Initialize MediaPipe components
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Error: Cannot open image '{image_path}'")

    # Initialize MediaPipe Hands
    with mp_hands.Hands(
        static_image_mode=True, max_num_hands=2, min_detection_confidence=0.3
    ) as hands:
        # Convert image to RGB and process
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_image)

        # Get image dimensions
        image_height, image_width, _ = image.shape

        # Create a copy for annotation
        annotated_image = image.copy()

        # Check for hand landmarks
        if results.multi_hand_landmarks:
            print(f"Detected {len(results.multi_hand_landmarks)} hand(s)")

            # Process each detected hand
            for i, (hand_landmarks, handedness) in enumerate(
                zip(results.multi_hand_landmarks, results.multi_handedness)
            ):
                # Print handedness information
                hand_label = handedness.classification[0].label
                hand_confidence = handedness.classification[0].score
                print(f"Hand {i+1}: {hand_label} (confidence: {hand_confidence:.2f})")

                # Draw landmarks
                mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

                # Print each landmark position
                print(f"\nLandmark positions for {hand_label} hand:")
                for landmark_id, landmark in enumerate(hand_landmarks.landmark):
                    # Get pixel coordinates
                    px = int(landmark.x * image_width)
                    py = int(landmark.y * image_height)
                    pz = landmark.z

                    # Print landmark information
                    landmark_name = mp_hands.HandLandmark(landmark_id).name
                    print(
                        f"  {landmark_id}. {landmark_name}: x={px}, y={py}, z={pz:.4f}"
                    )

                    # Add landmark ID text to the image
                    cv2.putText(
                        annotated_image,
                        str(landmark_id),
                        (px + 5, py + 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )
        else:
            print("No hands detected in the image")
            # Add a message to the image
            cv2.putText(
                annotated_image,
                "No hands detected",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        # Display the result only if requested
        if display_image:
            cv2.imshow("MediaPipe Hand Landmarks", annotated_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Save the output image if requested
        if save_output:
            if output_path is None:
                output_path = "hand_landmarks_output.jpg"

            # Make sure directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            cv2.imwrite(output_path, annotated_image)
            print(f"Output image saved to: {output_path}")

        # Return results
        return {
            "multi_hand_landmarks": results.multi_hand_landmarks,
            "multi_handedness": results.multi_handedness,
            "image_height": image_height,
            "image_width": image_width,
            "annotated_image": annotated_image,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process image with MediaPipe Hands")
    parser.add_argument(
        "-i",
        "--image_path",
        default="data/palm1.jpg",
        help="Path to input image file",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        default="output/mediapipe_hands.jpg",
        help="Path to save the annotated output image",
    )
    parser.add_argument(
        "-d",
        "--display",
        action="store_true",
        default=False,
        help="Display the output image using cv2.imshow (default: True)",
    )
    args = parser.parse_args()

    try:
        result = process_image(
            image_path=args.image_path,
            save_output=args.output_path is not None,
            output_path=args.output_path,
            display_image=args.display,
        )
        print(f"{result=}")
    except Exception as e:
        print(f"Error processing image: {str(e)}", file=sys.stderr)
        sys.exit(1)
