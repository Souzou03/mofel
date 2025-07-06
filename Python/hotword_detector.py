# hotword_detector.py

import json
from os.path import isfile
import numpy as np
from time import time as current_time_in_sec
import logging
from typing import Dict, Union

from eff_word_net.audio_processing import ModelRawBackend
from eff_word_net import RATE
from eff_word_net.audio_processing import MODEL_TYPE_MAPPER

class HotwordDetector:
    """
    EfficientWordベースのHotwordDetectorエンジン実装クラス
    """

    def __init__(
            self,
            hotword: str,
            model: ModelRawBackend,
            reference_file: str,
            threshold: float = 0.9,
            relaxation_time: float = 0.8,
            continuous: bool = True,
            verbose: bool = False):
        """
        HotwordDetectorインスタンスを初期化します。

        Args:
            hotword (str): ホットワード文字列
            model (ModelRawBackend): 使用するモデル
            reference_file (str): ホットワードの参照ファイルへのパス
            threshold (float): 一致と判断するための最小類似度スコア (0-1)
            relaxation_time (float): 一度検出してから次に検出するまでのクールダウンタイム（秒）
            continuous (bool): 連続的なストリームを処理するかどうか
            verbose (bool): 詳細なログを出力するかどうか
        """
        assert isfile(reference_file), "参照ファイルのパスが無効です"
        assert 0 < threshold < 1, "thresholdは0と1の間でなければなりません"

        with open(reference_file, 'r') as f:
            data = json.load(f)
        
        self.embeddings = np.array(data["embeddings"]).astype(np.float32)

        assert self.embeddings.shape[0] > 3, "最低でも4つのサンプルデータポイントが必要です"
        assert MODEL_TYPE_MAPPER[data["model_type"]] == type(model), "参照ファイルとモデルのタイプが一致しません"
        
        self.model = model
        self.hotword = hotword
        self.threshold = threshold
        self.continuous = continuous
        self.relaxation_time = relaxation_time
        self.verbose = verbose
        self.__last_activation_time = 0.0

    def __repr__(self) -> str:
        return f"Hotword: {self.hotword}"

    def scoreVector(self, inp_vec: np.ndarray) -> float:
        """入力ベクトルと参照エンベディングのスコアを計算し、relaxation timeを考慮します。"""
        score = self.model.scoreVector(inp_vec, self.embeddings)
        current_time = current_time_in_sec()

        if self.continuous:
            if score > self.threshold:
                if (current_time - self.__last_activation_time) < self.relaxation_time:
                    return 0.001  # relaxation time内のためスコアを低く返す

        if score > self.threshold:
            if self.verbose:
                print(f"Activation Gap for {self.hotword}:", current_time - self.__last_activation_time)
            self.__last_activation_time = current_time

        return score

    def scoreFrame(
            self,
            inp_audio_frame: np.ndarray,
            unsafe: bool = False
    ) -> Union[Dict[str, Union[bool, float]], None]:
        """
        音声フレームをベクトルに変換し、類似度をチェックします。

        Args:
            inp_audio_frame (np.ndarray): 1秒、16000Hzサンプリングの音声フレーム
            unsafe (bool): Falseの場合、無音や連続音を処理しないようにし、誤検出を減らします。

        Returns:
            Union[Dict[str, Union[bool, float]], None]: 
            検出結果の辞書、または音声アクティビティがない場合はNone。
            e.g. {"match": True, "confidence": 0.95}
        """
        if not unsafe:
            # 音声の開始部分が急激に大きくないかチェックし、誤検出を防ぐ
            if np.max(inp_audio_frame) == 0: return None
            normalized_frame = inp_audio_frame / np.max(inp_audio_frame)
            upperPoint = np.max(normalized_frame[:RATE // 10])
            if upperPoint > 0.2:
                return None

        inp_vec = self.model.audioToVector(inp_audio_frame)
        score = self.scoreVector(inp_vec)

        return {
            "match": score >= self.threshold,
            "confidence": score
        }
