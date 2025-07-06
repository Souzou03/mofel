# multi_hotword_detector.py

import numpy as np
from typing import Tuple, List, Optional

from eff_word_net.audio_processing import ModelRawBackend
from hotword_detector import HotwordDetector

# Type Aliases
HotwordDetectorArray = List[HotwordDetector]
MatchInfo = Tuple[HotwordDetector, float]
MatchInfoArray = List[MatchInfo]

class MultiHotwordDetector:
    """
    複数のHotwordDetectorを効率的に扱うためのラッパークラス
    """

    def __init__(
        self,
        detector_collection: HotwordDetectorArray,
        model: ModelRawBackend,
        continuous: bool = True
    ):
        """
        Args:
            detector_collection (HotwordDetectorArray): HotwordDetectorインスタンスのリスト
            model (ModelRawBackend): 使用するモデル
            continuous (bool): 連続的なストリームを処理するかどうか
        """
        assert len(detector_collection) > 1, "最低でも2つのHotwordDetectorインスタンスを渡してください"
        for detector in detector_collection:
            assert isinstance(detector, HotwordDetector), "リストにはHotwordDetectorインスタンスのみ含めることができます"

        self.model = model
        self.detector_collection = detector_collection
        self.continuous = continuous

    def findBestMatch(
        self,
        inp_audio_frame: np.ndarray,
        unsafe: bool = False
    ) -> Tuple[Optional[HotwordDetector], float]:
        """
        与えられた音声フレームに対して、最もスコアの高いホットワードを返します。
        
        Returns:
            Tuple[Optional[HotwordDetector], float]: 最も一致したDetectorとそのスコア。見つからなければ (None, 0.0)。
        """
        embedding = self.model.audioToVector(inp_audio_frame)

        best_match_detector: Optional[HotwordDetector] = None
        best_match_score: float = 0.0

        for detector in self.detector_collection:
            score = detector.scoreVector(embedding)

            if score < detector.threshold:
                continue

            if score > best_match_score:
                best_match_score = score
                best_match_detector = detector
        
        return best_match_detector, best_match_score

    def findAllMatches(
        self,
        inp_audio_frame: np.ndarray,
        unsafe: bool = False
    ) -> MatchInfoArray:
        """
        閾値を超えたすべてのホットワードをスコア順（降順）のリストで返します。
        
        Returns:
            MatchInfoArray: マッチした (Detector, score) のタプルのリスト
        """
        embedding = self.model.audioToVector(inp_audio_frame)
        matches: MatchInfoArray = []

        for detector in self.detector_collection:
            # Note: 元のコードの getMatchScoreVector は存在しないため scoreVector に修正
            score = detector.scoreVector(embedding)
            if score >= detector.threshold:
                matches.append((detector, score))

        # スコアの高い順にソート
        matches.sort(key=lambda item: item[1], reverse=True)
        return matches
