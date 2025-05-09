"""
Office-Specific Scenarios
five scenarios where a mobile robot performs simple but useful office tasks while reacting naturally to human behavior and space
"""
def check_office_task_interaction():
    # Simulated inputs
    task = get_current_task()                             # e.g., "make_coffee", "pickup_folder", etc.
    person_present = detect_person_near("Person A")
    folder_detected = detect_object_on_table("folder")
    coffee_machine_ready = check_coffee_machine_status()
    voice_command = detect_voice_command()                # "bring coffee", "take this", etc.
    object_received = check_object_received()
    delivery_location_clear = check_drop_zone()
    cup_held_by_robot = check_if_holding("cup")

    # Scenario 1: Make coffee and deliver
    if task == "make_coffee" and coffee_machine_ready:
        say("Making your coffee. One moment, please.")
        prepare_coffee()
        say("Coffee is ready. Heading your way.")
        deliver_to_person("Person B")
        say("Here is your coffee!")
        release_item("cup")
        continue_navigation()

    # Scenario 2: Pick up folder from Person A’s table and deliver to Document Hub
    elif task == "pickup_folder" and person_present and folder_detected:
        say("Picking up the folder from your desk.")
        grab_object("folder")
        say("Delivering it to the document hub.")
        navigate_to("Document Hub")
        place_item("folder")
        say("Folder delivered.")
        continue_navigation()

    # Scenario 3: Voice command “take this” when person hands an item
    elif voice_command == "take this" and object_received:
        say("Got it. Where should I bring it?")
        # Await new instruction or use default destination
        navigate_to("Reception")
        place_item("document")
        say("Item delivered.")
        continue_navigation()

    # Scenario 4: Blocked hallway with humans chatting
    elif detect_group_conversation() and detect_narrow_path():
        stop_movement()
        say("Hi there! May I pass through?")
        signal_presence()
        wait_until_path_clear(timeout=7)
        say("Thanks a lot.")
        continue_navigation()

    # Scenario 5: Deliver coffee but person not at desk
    elif task == "deliver_coffee" and not detect_person_near("Person C") and cup_held_by_robot:
        say("Hmm, Person C is not here.")
        navigate_to("Kitchen Station")
        say("Leaving the coffee at the kitchen station.")
        place_item("cup")
        continue_navigation()

    else:
        continue_navigation()
