import threading
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from pathlib import Path
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage


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
        self.screen_width = self.window.winfo_screenwidth()
        self.screen_height = self.window.winfo_screenheight()

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
        image_image_1 = PhotoImage(file=self.relative_to_assets("image_1.png"))
        image_1 = self.canvas.create_image(190.0, 127.0, image=image_image_1)
        self.image_refs.append(image_image_1)

        image_image_2 = PhotoImage(file=self.relative_to_assets("image_2.png"))
        image_2 = self.canvas.create_image(
            self.screen_width / 2, self.screen_height / 2, image=image_image_2
        )
        self.image_refs.append(image_image_2)

        # Create button with hover effect
        button_image_1 = PhotoImage(file=self.relative_to_assets("button_1.png"))
        button_image_hover_1 = PhotoImage(
            file=self.relative_to_assets("button_hover_1.png")
        )

        button_1 = Button(
            image=button_image_1,
            borderwidth=0,
            highlightthickness=0,
            command=self.on_stop_button_press,
            relief="flat",
        )
        button_1.place(
            x=(self.screen_width - 256) / 2,
            y=self.screen_height - 150,
            width=256.0,
            height=91.0,
        )

        # Store button and image references
        self.image_refs.extend([button_image_1, button_image_hover_1])
        self.button_refs.append(button_1)

        # Set up hover effect
        def button_1_hover(e):
            button_1.config(image=button_image_hover_1)

        def button_1_leave(e):
            button_1.config(image=button_image_1)

        button_1.bind("<Enter>", button_1_hover)
        button_1.bind("<Leave>", button_1_leave)
        print("Stop screen created.")

    def create_going_screen(self):
        """Create the going screen"""
        print("Creating going screen...")
        self.canvas.delete("all")
        self.image_refs = []  # Clear previous references
        self.button_refs = []

        self.canvas.create_text(
            self.screen_width / 2,
            self.screen_height / 2,
            text="GOING",
            fill="green",
            font=("Helvetica", 50, "bold"),
        )
        print("Going screen created.")

    def create_waiting_screen(self):
        """Create the waiting screen"""
        print("Creating waiting screen...")
        self.canvas.delete("all")
        self.image_refs = []  # Clear previous references
        self.button_refs = []

        self.canvas.create_text(
            self.screen_width / 2,
            self.screen_height / 2,
            text="WAITING",
            fill="blue",
            font=("Helvetica", 60, "bold"),
        )
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
            user_input = input("Enter screen command (1: Stop, 2: Going, 3: Waiting): ")
            if user_input == "1":
                self.create_stop_screen()
            elif user_input == "2":
                self.create_going_screen()
            elif user_input == "3":
                self.create_waiting_screen()
            else:
                print("Invalid input. Please enter 1, 2, or 3.")


def main(args=None):
    print("Initializing ROS 2...")
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
        rclpy.shutdown()
        print("ROS 2 shutdown complete.")


if __name__ == "__main__":
    main()
