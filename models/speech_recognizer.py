import rclpy
from rclpy.node import Node
import http.client as httplib
import speech_recognition as sr
from std_msgs.msg import String

class SpeechRecognizer(Node):
    def __init__(self):
        super().__init__("speech_recognizer")
        self.publisher_ = self.create_publisher(String, "speech_recognizer", 10)
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

        self.get_logger().info("Adjusting for ambient noise...")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)

        self.timer = self.create_timer(0.1, self.recognize_speech)

    def check_internet_connection(self):
        connection = httplib.HTTPConnection("www.google.com", timeout=5)
        try:
            connection.request("HEAD", "/")
            connection.close()
            return True
        except:
            connection.close()
            return False

    def recognize_speech(self):
        self.get_logger().info('Listening...')
        try:
            with self.microphone as source:
                audio = self.recognizer.listen(source)
            self.get_logger().info('Got a sound; recognizing...')

            recognized_speech = ""
            if self.check_internet_connection():
                try:
                    recognized_speech = self.recognizer.recognize_google(audio)
                except sr.UnknownValueError:
                    self.get_logger().error("Could not understand audio.")
                except sr.RequestError:
                    self.get_logger().error("Could not request results.")
            else:
                self.get_logger().error("No internet connection. Unable to use Google Speech Recognition.")

            if recognized_speech:
                self.get_logger().info("You said: " + recognized_speech)
                msg = String()
                msg.data = recognized_speech
                self.publisher_.publish(msg)

        except Exception as exc:
            self.get_logger().error(f"Error: {exc}")

def main(args=None):
    rclpy.init(args=args)
    speech_recognizer = SpeechRecognizer()
    try:
        rclpy.spin(speech_recognizer)
    except KeyboardInterrupt:
        pass
    finally:
        speech_recognizer.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
