# Speech-and-Text

语音转文字（支持实时麦克风输入和从音频文件读入）：

- 百度 API

文字转语音：

- pyttsx3

&nbsp;

## Environment

- Python 3.6.7
- MacOS（以下环境配置方式均基于Mac系统，其他系统的配置方式可能会有一些不同）

&nbsp;

## Speech to Text

### 百度

在 https://cloud.baidu.com/product/speech 申请API。

文档：http://ai.baidu.com/docs#/ASR-API



#### Configuration

安装：

```python
pip install baidu-aip
```

在 `speech2text_baidu()` 中填入APPID、API_KEY、SECRET_KEY：

```python
APP_ID = ""
API_KEY = ""
SECRET_KEY = ""
```

(也可以直接使用REST API：[Demo](https://github.com/Baidu-AIP/speech-demo)



#### Usage

```python
from speech2text import speech2text_baidu
# 从文件读入
speech2text_baidu(audio_path = "path_of_audio", if_microphone = False)
# 从麦克风读入
speech2text_baidu(if_microphone = True)
```



&nbsp;

### SpeechRecognition

使用了Python的语音识别库 [SpeechRecognition](https://pypi.org/project/SpeechRecognition/)

源码：https://github.com/Uberi/speech_recognition

&nbsp;

#### Configuration

##### SpeechRecognition

安装：

```python
pip install SpeechRecognition
```



##### PyAudio

使用麦克风进行输入

主页：http://people.csail.mit.edu/hubert/pyaudio/

```python
# Mac上的安装方式

xcode-select --install	# 安装xcode, 已经装好的的话，执行的时候会提示

# 先用homebrew安装portaudio（pyaudio需要的库），否则会提示：'portaudio.h' file not found
brew remove portaudio	# 先用homebrew卸载
brew install portaudio	# 重新安装

sudo pip install pyaudio	# 安装pyaudio
```

Reference: https://stackoverflow.com/questions/33851379/pyaudio-installation-on-mac-python-3


&nbsp;

## Text to Speech

使用了Python的文字转语音库 [pyttsx3](https://pypi.org/project/pyttsx3/)

源码：https://github.com/nateshmbhat/pyttsx3

文档：https://pyttsx3.readthedocs.io



### Configuration

```python
pip install pyttsx3
pip install pyobjc # 依赖模块
```



### Usage

```python
from speech2text import text_to_speech
# Example
text_to_speech(sentence = "人类的本质是复读机")
```
