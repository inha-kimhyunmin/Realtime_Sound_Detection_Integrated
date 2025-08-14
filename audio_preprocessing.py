import numpy as np
import sounddevice as sd
import time
import os
import json
from scipy import signal


def get_audio_volume(audio):
    """오디오의 RMS 볼륨과 최대 절댓값 계산"""
    rms = np.sqrt(np.mean(audio**2))
    max_val = np.max(np.abs(audio))
    return rms, max_val

def detect_clipping(audio, threshold=0.95):
    """오디오 클리핑 감지"""
    clipped_samples = np.sum(np.abs(audio) >= threshold)
    clipping_ratio = clipped_samples / len(audio)
    return clipping_ratio > 0.01  # 1% 이상 클리핑되면 True

def apply_compressor(audio, threshold=0.7, ratio=4.0, attack=0.003, release=0.1, sample_rate=16000):
    """간단한 컴프레서 적용 (과도한 볼륨 제어)"""
    # 간단한 피크 제한 컴프레서
    compressed = audio.copy()
    
    # 임계값을 넘는 부분을 압축
    mask = np.abs(audio) > threshold
    if np.any(mask):
        # 압축 비율 적용
        compressed[mask] = np.sign(audio[mask]) * (
            threshold + (np.abs(audio[mask]) - threshold) / ratio
        )
    
    return compressed

def normalize_audio_adaptive(audio, target_rms=0.1, max_gain=3.0):
    """적응형 오디오 정규화"""
    current_rms = np.sqrt(np.mean(audio**2))
    
    if current_rms < 1e-6:  # 거의 무음인 경우
        return audio
    
    # 목표 RMS에 맞춰 게인 계산
    gain = target_rms / current_rms
    
    # 최대 게인 제한
    gain = min(gain, max_gain)
    
    normalized = audio * gain
    
    # 최종 클리핑 방지
    max_val = np.max(np.abs(normalized))
    if max_val > 0.95:
        normalized = normalized * (0.95 / max_val)
    
    return normalized

def preprocess_audio(audio, sample_rate=16000):
    """통합 오디오 전처리 함수"""
    original_rms, original_max = get_audio_volume(audio)
    
    # 1. 클리핑 감지
    is_clipped = detect_clipping(audio)
    
    # 2. 컴프레서 적용 (과도한 볼륨 제어)
    if original_max > 0.8 or is_clipped:
        audio = apply_compressor(audio, threshold=0.6, ratio=6.0)
    
    # 4. 최종 안전장치 (하드 리미터)
    audio = np.clip(audio, -0.95, 0.95)
    
    # 전처리 정보 반환
    processed_rms, processed_max = get_audio_volume(audio)
    preprocessing_info = {
        'original_rms': original_rms,
        'original_max': original_max,
        'processed_rms': processed_rms,
        'processed_max': processed_max,
        'was_clipped': is_clipped,
        'volume_reduced': original_rms > processed_rms * 1.5
    }
    
    return audio, preprocessing_info