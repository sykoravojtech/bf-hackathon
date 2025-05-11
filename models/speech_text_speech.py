import pyttsx3
import speech_recognition as sr
from vosk import Model, KaldiRecognizer
import sounddevice as sd
import queue
import json

def robot_pipeline(input_mode="text", offline=False, model_path="/Users/kt/Desktop/Hackathon/humor_io/bf-hackathon/models/vosk-model-small-en-us-0.15"):
    # Define commands
    commands = {
        "turn left": "robot is turning left",
        "turn right": "robot is turning right",
        "stop": "robot has stopped",
        "wait": "robot is waiting for a moment",
        "move forward": "robot is moving forward",
        "move backward": "robot is moving backward",
        "start": "robot is starting",
        "pause": "robot is paused",
        "go to sleep": "robot is sleeping",
        "wake up": "robot is waking up",
        "bring me beer": "beer coming up",
        "bring me coffee": "coffee coming up",
        "move right": "robot is moving to the right",
        "move left": "robot is moving to the left",
        "can you dance": "Of course, if you show me how to dance"
    }

    # TTS engine
    tts_engine = pyttsx3.init()

    def respond(text):
        print(f"🤖 {text}")
        tts_engine.say(text)
        tts_engine.runAndWait()

    def process_command(text):
        for keyword, response in commands.items():
            if keyword in text:
                return response
        return "Sorry, I did not understand the command."

    def listen_command_online():
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("🎤 Listening (online)...")
            audio = recognizer.listen(source)
        try:
            return recognizer.recognize_google(audio).lower()
        except:
            return "Speech recognition failed (online)."

    def listen_command_offline_vosk():
        q = queue.Queue()
        model = Model(model_path)
        rec = KaldiRecognizer(model, 16000)

        def callback(indata, frames, time, status):
            if status:
                print(status)
            q.put(bytes(indata))

        print("🎤 Listening (offline)...")
        with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                               channels=1, callback=callback):
            while True:
                data = q.get()
                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    return result.get("text", "").lower()

    if input_mode == "text":
        user_input = input("⌨️ Type your command: ").lower()
        print(f"🗣️ You typed: {user_input}")  # Show typed input
        response = f"Got it, you said: {user_input}"  # The robot repeats back the input
    elif input_mode == "speech":
        if offline:
            user_input = listen_command_offline_vosk()
        else:
            user_input = listen_command_online()
        print(f"🗣️ You said: {user_input}")
        response = process_command(user_input)  # Process speech command to a response
    else:
        respond("Invalid input mode.")
        return

    print(f"🤖 Response: {response}")  # Print the response
    respond(response)  # Speak the response
