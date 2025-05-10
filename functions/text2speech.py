from gtts import gTTS
import os

def text_to_speech(text, lang='en'):
    tts = gTTS(text=text, lang=lang)
    tts.save("output.mp3")
    os.system("mpg321 output.mp3")  # You can use another player if mpg321 is not available

if __name__ == "__main__":
    text = "Hello, this is a text to speech test."
    text_to_speech(text)