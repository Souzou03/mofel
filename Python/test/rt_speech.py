import speech_recognition as sr

# Recognizerオブジェクトを初期化
r = sr.Recognizer()

# 発話の終了と見なす無音の秒数（値を調整可能）
r.pause_threshold = 1.0 

with sr.Microphone() as source:
    print("ノイズレベルを調整します。1秒ほど静かにしてください...")
    # ノイズレベルを自動調整
    r.adjust_for_ambient_noise(source,duration=0.5)
    print("準備ができました。話しかけてください。")

    try:
        # プログラム開始後、最初の発話を待機して聞き取る
        audio = r.listen(source)
        
        print("音声認識を実行中...")
        
        # GoogleのAPIで音声をテキストに変換
        text = r.recognize_google(audio, language='ja-JP')
        print("----------------------------------------")
        print("文字起こし結果: " + text)
        print("----------------------------------------")

    except sr.UnknownValueError:
        print("音声を認識できませんでした。")
    except sr.RequestError as e:
        print(f"APIにリクエストできませんでした; {e}")

print("プログラムを終了します。")
