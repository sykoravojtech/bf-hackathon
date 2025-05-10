from gtts import gTTS
from playsound import playsound

def text_to_speech(text, lang='en'):
    tts = gTTS(text=text, lang=lang)
    tts.save("output.mp3")
    playsound("output.mp3")  # Play the audio immediately

if __name__ == "__main__":
    text = input("Enter the text to convert to speech: ")  # Prompt for text input
    text_to_speech(text)