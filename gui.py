import threading
from pathlib import Path
from tkinter import Button, Canvas, Entry, PhotoImage, Text, Tk

# Add try/except block for ROS2 imports to handle version differences
try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String

    ROS_AVAILABLE = True
except ImportError:
    print("WARNING: ROS2 packages not found. Running in standalone mode.")
    ROS_AVAILABLE = False

    # Create dummy classes for non-ROS environment
    class Node:
        def __init__(self, name):
            self.name = name

        def create_subscription(self, *args, **kwargs):
            return None

        def create_publisher(self, *args, **kwargs):
            return None

        def get_logger(self):
            return DummyLogger()

        def destroy_node(self):
            pass

    class DummyLogger:
        def info(self, msg):
            print(f"INFO: {msg}")

        def warn(self, msg):
            print(f"WARN: {msg}")

    class String:
        def __init__(self, data=""):
            self.data = data


class GUI(Node):
    ASSETS_PATH = "build/assets/frame0"

    def __init__(self):
        """Initialize the GUI window and ROS 2 node"""
        print("Initializing GUI Node...")
        super().__init__("gui_node")

        # ROS 2 publishers and subscribers
        print("Setting up ROS 2 publishers and subscribers...")
        self.screen_subscriber = self.create_subscription(
            String, "screen_command", self.screen_callback, 10
        )
        self.button_publisher = self.create_publisher(String, "button_press", 10)

        # Initialize GUI
        print("Initializing GUI window...")
        self.window = Tk()

        # Get screen dimensions
        # self.screen_width = self.window.winfo_screenwidth()
        # self.screen_height = self.window.winfo_screenheight()
        # Set fixed screen dimensions to 1920x1080
        self.screen_width = 1920
        self.screen_height = 1080

        self.window.geometry(f"{self.screen_width}x{self.screen_height}")
        self.window.configure(bg="#2C2C2C")

        self.canvas = Canvas(
            self.window,
            bg="#2C2C2C",
            height=self.screen_height,
            width=self.screen_width,
            bd=0,
            highlightthickness=0,
            relief="ridge",
        )

        self.canvas.place(x=0, y=0)
        self.window.resizable(True, True)

        # Store references to prevent garbage collection
        self.image_refs = []
        self.button_refs = []

        self.create_stop_screen()
        print("GUI Node initialized successfully.")

    def relative_to_assets(self, path: str) -> Path:
        """Helper function to resolve asset paths"""
        print(f"Resolving asset path for: {path}")
        return self.ASSETS_PATH / Path(path)

    def create_stop_screen(self):
        """Create the stop screen"""
        print("Creating stop screen...")
        self.canvas.delete("all")
        self.image_refs = []  # Clear previous references
        self.button_refs = []

        # Load and place images
        image_image_1 = PhotoImage(file=self.relative_to_assets("image_sew.png"))
        image_1 = self.canvas.create_image(190.0, 127.0, image=image_image_1)
        self.image_refs.append(image_image_1)

        # Get main image dimensions for proper centering
        image_image_2 = PhotoImage(file=self.relative_to_assets("image_stop.png"))
        # Use exact center of screen (no decimal points to avoid subpixel rendering issues)
        center_x = self.screen_width // 2
        center_y = self.screen_height // 2

        image_2 = self.canvas.create_image(center_x, center_y, image=image_image_2)
        self.image_refs.append(image_image_2)

        # Create button with hover effect
        button_image_1 = PhotoImage(file=self.relative_to_assets("button_1.png"))
        button_image_hover_1 = PhotoImage(
            file=self.relative_to_assets("button_hover_1.png")
        )

        # Get actual button image dimensions
        button_width = button_image_1.width()
        button_height = button_image_1.height()

        # Calculate position for the button - use exact same center x as the image
        button_y = self.screen_height - 150

        # Add offset to move button slightly to the left
        button_offset_x = -20  # Negative value moves left, positive moves right

        # Create a pure canvas-based button using the same center_x as the main image with offset
        button_1 = self.canvas.create_image(
            center_x + button_offset_x,  # Apply the offset here
            button_y + (button_height / 2),  # y position (center of image)
            image=button_image_1,
            tags=("button_1",),
        )

        # Calculate hitbox for button click detection (used in event handler)
        button_x = (
            center_x + button_offset_x - (button_width / 2)
        )  # Left edge of button

        # Store button and image references
        self.image_refs.extend([button_image_1, button_image_hover_1])

        # Define event handlers for the canvas-based button
        def on_button_enter(event):
            self.canvas.itemconfig("button_1", image=button_image_hover_1)

        def on_button_leave(event):
            self.canvas.itemconfig("button_1", image=button_image_1)

        def on_button_press(event):
            # Check if click is within the button area
            if (
                button_x <= event.x <= button_x + button_width
                and button_y <= event.y <= button_y + button_height
            ):
                self.on_stop_button_press()

        # Bind events to the canvas
        self.canvas.tag_bind("button_1", "<Enter>", on_button_enter)
        self.canvas.tag_bind("button_1", "<Leave>", on_button_leave)
        self.canvas.bind("<Button-1>", on_button_press)

        print("Stop screen created.")

    def create_going_screen(self):
        """Create the going screen"""
        print("Creating going screen...")
        self.canvas.delete("all")
        self.image_refs = []  # Clear previous references
        self.button_refs = []

        # Load and place images
        image_image_1 = PhotoImage(file=self.relative_to_assets("image_sew.png"))
        image_1 = self.canvas.create_image(190.0, 127.0, image=image_image_1)
        self.image_refs.append(image_image_1)

        # Get main image dimensions for proper centering
        image_image_2 = PhotoImage(file=self.relative_to_assets("image_go.png"))
        # Use exact center of screen (no decimal points to avoid subpixel rendering issues)
        center_x = self.screen_width // 2
        center_y = self.screen_height // 2

        image_2 = self.canvas.create_image(center_x, center_y, image=image_image_2)
        self.image_refs.append(image_image_2)

        print("Going screen created.")

    def create_waiting_screen(self):
        """Create the waiting screen"""
        print("Creating waiting screen...")
        self.canvas.delete("all")
        self.image_refs = []  # Clear previous references
        self.button_refs = []

        # Load and place images
        image_image_1 = PhotoImage(file=self.relative_to_assets("image_sew.png"))
        image_1 = self.canvas.create_image(190.0, 127.0, image=image_image_1)
        self.image_refs.append(image_image_1)

        # Get main image dimensions for proper centering
        image_image_2 = PhotoImage(file=self.relative_to_assets("image_wait.png"))
        # Use exact center of screen (no decimal points to avoid subpixel rendering issues)
        center_x = self.screen_width // 2
        center_y = self.screen_height // 2

        image_2 = self.canvas.create_image(center_x, center_y, image=image_image_2)
        self.image_refs.append(image_image_2)

        # Create button with hover effect
        button_image_1 = PhotoImage(file=self.relative_to_assets("button_1.png"))
        button_image_hover_1 = PhotoImage(
            file=self.relative_to_assets("button_hover_1.png")
        )

        # Get actual button image dimensions
        button_width = button_image_1.width()
        button_height = button_image_1.height()

        # Calculate position for the button - use exact same center x as the image
        button_y = self.screen_height - 150

        # Add offset to move button slightly to the left
        button_offset_x = -20  # Negative value moves left, positive moves right

        # Create a pure canvas-based button using the same center_x as the main image with offset
        button_1 = self.canvas.create_image(
            center_x + button_offset_x,  # Apply the offset here
            button_y + (button_height / 2),  # y position (center of image)
            image=button_image_1,
            tags=("button_1",),
        )

        # Calculate hitbox for button click detection (used in event handler)
        button_x = (
            center_x + button_offset_x - (button_width / 2)
        )  # Left edge of button

        # Store button and image references
        self.image_refs.extend([button_image_1, button_image_hover_1])

        # Define event handlers for the canvas-based button
        def on_button_enter(event):
            self.canvas.itemconfig("button_1", image=button_image_hover_1)

        def on_button_leave(event):
            self.canvas.itemconfig("button_1", image=button_image_1)

        def on_button_press(event):
            # Check if click is within the button area
            if (
                button_x <= event.x <= button_x + button_width
                and button_y <= event.y <= button_y + button_height
            ):
                self.on_stop_button_press()

        # Bind events to the canvas
        self.canvas.tag_bind("button_1", "<Enter>", on_button_enter)
        self.canvas.tag_bind("button_1", "<Leave>", on_button_leave)
        self.canvas.bind("<Button-1>", on_button_press)

        print("Waiting screen created.")

    def screen_callback(self, msg: String):
        """Callback to handle screen change commands"""
        print(f"Received screen command: {msg.data}")
        command = msg.data.lower()
        if command == "stop":
            self.create_stop_screen()
        elif command == "going":
            self.create_going_screen()
        elif command == "waiting":
            self.create_waiting_screen()
        else:
            self.get_logger().warn(f"Unknown command: {command}")

    def on_stop_button_press(self):
        """Handle stop button press"""
        print("Stop button pressed.")
        self.get_logger().info("Stop button pressed")
        self.button_publisher.publish(String(data="stop_button_pressed"))

    def run(self):
        """Start the main loop of the GUI"""
        print("Starting GUI main loop...")
        self.get_logger().info("Starting GUI...")
        self.window.mainloop()

    def handle_terminal_input(self):
        """Handle terminal input to change screens"""
        while True:
            user_input = input(
                "Enter screen command (1: Stop, 2: Going, 3: Waiting, exit or quit): "
            )
            print(f"{user_input=}")
            if user_input == "1":
                self.create_stop_screen()
            elif user_input == "2":
                self.create_going_screen()
            elif user_input == "3":
                self.create_waiting_screen()
            # elif user_input.lower() in ["exit", "quit"]:
            #     break
            else:
                print("Invalid input. Please enter 1, 2, or 3.")


def main(args=None):
    print("Initializing ROS 2...")
    if ROS_AVAILABLE:
        rclpy.init(args=args)
    gui = GUI()

    # Start a thread for handling terminal input
    input_thread = threading.Thread(target=gui.handle_terminal_input, daemon=True)
    input_thread.start()

    try:
        gui.run()
    except KeyboardInterrupt:
        print("Shutting down GUI...")
        gui.get_logger().info("Shutting down GUI...")
    finally:
        gui.destroy_node()
        if ROS_AVAILABLE:
            rclpy.shutdown()
        print("ROS 2 shutdown complete.")


if __name__ == "__main__":
    main()
