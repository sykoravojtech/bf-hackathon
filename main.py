import time
from typing import Callable

from sensors.display import show_display
from sensors.moving import rotate, set_velocity
from sensors.speaker import speaker_say


class Controller:
    def __init__(
        self,
        speaker_say: Callable,
        rotate: Callable,
        set_velocity: Callable,
        show_display: Callable,
    ):
        self.robot_on = True
        self.speaker_say = speaker_say
        self.rotate = rotate
        self.set_velocity = set_velocity
        self.show_display = show_display

    def scenario_check_loop(self, authometry: dict, camera, lidar, microphone, touch):
        """
        Checks all scenarios. If one is happening run it. If none is happening just continue.
        """
        print("Checking every scenario")

        try:
            user_input = input("Enter a number (or 'quit' to exit): ")
            if user_input.lower() == "quit":
                self.robot_on = False
                self.speaker_say("Shutting down")
                return

            try:
                number = int(user_input)
                print(f"You entered: {number}")
            except ValueError:
                print("Invalid input. Please enter a number or 'quit'")
                self.speaker_say("Invalid input, please try again")
        except KeyboardInterrupt:
            self.robot_on = False
            self.speaker_say("Shutting down")


def main():
    print("Starting Program")

    # initialize all thats needed

    # All possible inputs into our program
    authometry = {"position": None, "direction": None, "speed": None}
    camera = None
    lidar = None  # 3D point cloud
    microphone = None  # voice mp3 input?
    touch: int = (
        None  # each button has an int assigned so for example 1:turn_left, 2:turn_right
    )

    # All possible outputs of our program
    speaker_fn = speaker_say
    rotate_fn = rotate
    velocity_fn = set_velocity
    display_fn = show_display

    controller = Controller(
        speaker_say=speaker_fn,
        rotate=rotate_fn,
        set_velocity=velocity_fn,
        show_display=display_fn,
    )

    # Check every second if any scenario is happening
    while controller.robot_on:
        controller.scenario_check_loop(authometry, camera, lidar, microphone, touch)
        time.sleep(1)


if __name__ == "__main__":
    main()
