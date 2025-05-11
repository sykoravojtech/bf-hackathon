import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from pathlib import Path
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage


class GUI(Node):
    ASSETS_PATH = "build/assets/frame0"

    def __init__(self):
        """Initialize the GUI window and ROS 2 node"""
        super().__init__("gui_node")

        # ROS 2 publishers and subscribers
        self.screen_subscriber = self.create_subscription(
            String, "screen_command", self.screen_callback, 10
        )
        self.button_publisher = self.create_publisher(String, "button_press", 10)

        # Initialize GUI
        self.window = Tk()
        self.window.geometry("1850x1080")
        self.window.configure(bg="#2C2C2C")

        self.canvas = Canvas(
            self.window,
            bg="#2C2C2C",
            height=1080,
            width=1800,
            bd=0,
            highlightthickness=0,
            relief="ridge",
        )

        self.canvas.place(x=0, y=0)
        self.window.resizable(False, False)

        # Store references to prevent garbage collection
        self.image_refs = []
        self.button_refs = []

    def relative_to_assets(self, path: str) -> Path:
        """Helper function to resolve asset paths"""
        return self.ASSETS_PATH / Path(path)

    def create_stop_screen(self):
        """Create the stop screen"""
        self.canvas.delete("all")
        self.image_refs = []  # Clear previous references
        self.button_refs = []

        # Load and place images
        image_image_1 = PhotoImage(file=self.relative_to_assets("image_1.png"))
        image_1 = self.canvas.create_image(190.0, 127.0, image=image_image_1)
        self.image_refs.append(image_image_1)

        image_image_2 = PhotoImage(file=self.relative_to_assets("image_2.png"))
        image_2 = self.canvas.create_image(974.0, 540.0, image=image_image_2)
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
        button_1.place(x=832.0, y=922.0, width=256.0, height=91.0)

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

    def create_going_screen(self):
        """Create the going screen"""
        self.canvas.delete("all")
        self.image_refs = []  # Clear previous references
        self.button_refs = []

        self.canvas.create_text(
            900,
            540,
            text="GOING",
            fill="green",
            font=("Helvetica", 50, "bold"),
        )

    def create_waiting_screen(self):
        """Create the waiting screen"""
        self.canvas.delete("all")
        self.image_refs = []  # Clear previous references
        self.button_refs = []

        self.canvas.create_text(
            900, 540, text="WAITING", fill="blue", font=("Helvetica", 60, "bold")
        )

    def screen_callback(self, msg: String):
        """Callback to handle screen change commands"""
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
        self.get_logger().info("Stop button pressed")
        self.button_publisher.publish(String(data="stop_button_pressed"))

    def run(self):
        """Start the main loop of the GUI"""
        self.get_logger().info("Starting GUI...")
        self.window.mainloop()


def main(args=None):
    rclpy.init(args=args)
    gui = GUI()

    try:
        gui.run()
    except KeyboardInterrupt:
        gui.get_logger().info("Shutting down GUI...")
    finally:
        gui.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
