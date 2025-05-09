"""

10 realistic and natural human-robot interaction scenarios that could happen in a factory or warehouse setting.

Each scenario:

Represents a unique interaction type (gesture, voice, proximity, obstacle, etc.).

Includes natural responses (speech, light, movement).

Is processed sequentially in the same function.

"""


def check_human_interaction():
    # Sensor input assumptions
    human_in_path = detect_human()
    distance_to_human = get_human_distance()
    hand_gesture = detect_gesture()  # "raised_hand", "point_left", "point_right", etc.
    voice_command = detect_voice_command()  # "hold", "stop", "go", "follow me", etc.
    in_narrow_space = check_narrow_space()
    obstacle_detected = detect_obstacle()
    touch_input = detect_touch()
    emergency_signal = detect_emergency()
    proximity_alert = detect_proximity()  # e.g., "very_close", "safe"
    light_signal_seen = detect_light_signal()  # e.g., "green_wave", "red_stop"
    pointing_direction = detect_pointing_direction()  # e.g., "left", "right", "none"

    # Scenario 1: Human crosses path
    if human_in_path and distance_to_human < 1.5:
        slow_down()
        say("Excuse me, passing through.")
        signal_presence()
        wait_until_path_clear(timeout=5)
        say("Thank you!")
        continue_navigation()

    # Scenario 2: Hand raised to pause
    elif hand_gesture == "raised_hand" and distance_to_human < 3.0:
        pause_movement(duration=3)
        say("Sure, pausing for you.")
        acknowledge_gesture()
        say("Resuming task now.")
        continue_navigation()

    # Scenario 3: Voice command "hold" in a narrow space
    elif voice_command == "hold" and in_narrow_space:
        stop_temporarily(duration=5)
        say("Holding position. Please go ahead.")
        say("Continuing my task now.")
        continue_navigation()

    # Scenario 4: Obstacle detected (e.g., dropped item or cart)
    elif obstacle_detected:
        stop_movement()
        say("Obstacle ahead. Waiting to proceed.")
        wait_until_path_clear(timeout=5)
        say("Path is clear. Resuming.")
        continue_navigation()

    # Scenario 5: Human touches the robot
    elif touch_input == "tap":
        pause_movement(duration=2)
        say("Hello! How can I help?")
        # Optional: await further input or timeout
        say("Going back to work now.")
        continue_navigation()

    # Scenario 6: Emergency signal detected (e.g., alarm, stop button)
    elif emergency_signal:
        stop_movement()
        say("Emergency detected. Stopping all tasks.")
        signal_alert()
        wait_for_manual_reset()
        # Do not resume automatically

    # Scenario 7: Proximity alert (someone stands too close too long)
    elif proximity_alert == "very_close":
        say("Please keep a safe distance.")
        back_up_slightly()
        continue_navigation()

    # Scenario 8: Light signal interaction (e.g., green wave = go, red = stop)
    elif light_signal_seen == "red_stop":
        stop_temporarily(duration=3)
        say("Stopping for visual signal.")
        continue_navigation()
    elif light_signal_seen == "green_wave":
        say("Received go signal.")
        continue_navigation()

    # Scenario 9: Pointing direction to redirect path
    elif hand_gesture in ["point_left", "point_right"]:
        new_direction = pointing_direction
        say(f"Changing path to the {new_direction}.")
        change_direction(new_direction)
        continue_navigation()

    # Scenario 10: Voice command “follow me”
    elif voice_command == "follow me":
        say("Okay, I’m following you.")
        start_follow_mode()
        # Follow until stop command or timeout
        say("Let me know when to stop.")
        # This may override normal task flow

    # Default behavior
    else:
        continue_navigation()
