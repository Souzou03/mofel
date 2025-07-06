import speech_recognition as sr
# SpeechRecognitionの設定
r = sr.Recognizer()
# 発話の終了と見なす無音の秒数
r.pause_threshold = 1.0 
#================================================

def listen():
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source,duration=0.5)
        try:
            audio = r.listen(source)
            print("音声認識を実行中...")
            text = r.recognize_google(audio, language='ja-JP')
            print("----------------------------------------")
            print("文字起こし結果: " + text)
            print("----------------------------------------")
            return text
        except sr.UnknownValueError:
            print("音声を認識できませんでした。")
            return None
        except sr.RequestError as e:
            print(f"APIにリクエストできませんでした; {e}")
            return None