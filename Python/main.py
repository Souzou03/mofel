# main.py

import os
from eff_word_net.streams import SimpleMicStream
from eff_word_net.audio_processing import Resnet50_Arc_loss

from hotword_detector import HotwordDetector
from multi_hotword_detector import MultiHotwordDetector


def wakeup():
    # 共通のベースモデルを定義
    base_model = Resnet50_Arc_loss()
    mofel = HotwordDetector(
        hotword="mofel",
        model=base_model,
        reference_file="./mofel/model/mofel_ref.json",
        threshold=0.7,
        relaxation_time=2
    )
    stop = HotwordDetector(
        hotword="stop",
        model=base_model,
        reference_file="./stop/model/stop_ref.json",
        threshold=0.7,
        relaxation_time=2
    )
    multi_hotword_detector = MultiHotwordDetector(
        detector_collection=[mofel, stop],
        model=base_model,
        continuous=True,
    )
    mic_stream = SimpleMicStream(
        window_length_secs=1.5, 
        sliding_window_secs=0.75
    )
    mic_stream.start_stream()

    print("ウェイクアップワード検出中:", " / ".join([d.hotword for d in multi_hotword_detector.detector_collection]))

    # 検出ループ
    while True:
        frame = mic_stream.getFrame()
        # 最もスコアの高いものを探す
        detector, score = multi_hotword_detector.findBestMatch(frame)
        
        # detectorがNoneでない場合（＝閾値を超える検出があった場合）に結果を表示
        if detector is not None:
            print(f"Detected: {detector.hotword}, Confidence: {score:.4f}")
            return detector.hotword


if __name__ == "__main__":
    while True:
        try:
            hotword = wakeup()
            print(f"ウェイクアップワード '{hotword}' が検出されました。")
        except KeyboardInterrupt:
            print("プログラムを終了します。")
            break
        except Exception as e:
            print(f"エラーが発生しました: {e}")

