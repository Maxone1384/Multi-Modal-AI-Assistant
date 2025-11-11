from gtts import gTTS

def text_to_speech(text, save_path="output.mp3"):
    tts = gTTS(text=text, lang='fa')
    tts.save(save_path)
    return save_path
