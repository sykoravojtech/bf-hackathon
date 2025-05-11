from gtts import gTTS
from playsound import playsound

def text_to_speech(text, lang='en'):
    tts = gTTS(text=text, lang=lang)
    tts.save("output.mp3")
    playsound("output.mp3")  # Play the audio immediately

if __name__ == "__main__":
    text = "Hello, welcome to the text-to-speech demo!"  # Provide text directly
    text_to_speech(text)