# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import yaml

import speech_recognition as sr
from pydub import AudioSegment
# Baidu API
from aip import AipSpeech
#import pyttsx3

# init SpeechRecognition object
r = sr.Recognizer()

def _record():
    """Get input audio from microphone, and write to audio file."""
    with sr.Microphone() as source:
        # 校准环境噪声水平的energy threshold
        r.adjust_for_ambient_noise(source, duration = 1)
        audio = r.listen(source, timeout = 3, phrase_time_limit = 2)

    file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "audio/speech.wav")
    with open(file_name, "wb") as f:
        f.write(audio.get_wav_data())
    
    return _get_file_content(file_name)

def _get_file_content(file_name):
    """Get speech from audio file."""
    speech = AudioSegment.from_wav(file_name).set_frame_rate(16000)
    return speech.raw_data

def speech2text_baidu(audio_path: str="test.wav", if_microphone: bool=True):
    """Baidu ASR API."""
    # get API info from https://cloud.baidu.com/product/speech
    config_file = os.path.join(os.path.dirname(__file__), 'config.yml')
    with open(config_file) as f:
        config = yaml.load(f)
    app_id = str(config['app_id'])
    api_key = config['api_key']
    secret_key = config['secret_key']
    client = AipSpeech(app_id, api_key, secret_key)

    # input from microphone
    if if_microphone:
        result = client.asr(
            _record(),
            'pcm',
            16000,
            # recognize Mandarin
            {'dev_pid': 1537}, 
        )
    # input from file
    else:
        result = client.asr(
            _get_file_content(audio_path),
            'pcm',
            16000,
            {'dev_pid': 1537},
        )

    if result["err_msg"] != "success.":
        return "..."
    else:
        return result['result'][0]


#def text_to_speech(sentence: str):
#    engine = pyttsx3.init()
#    engine.say(sentence)
#    engine.runAndWait()

