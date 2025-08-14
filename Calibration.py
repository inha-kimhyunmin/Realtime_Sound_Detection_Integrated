import sounddevice as sd
import numpy as np
import json
import threading
from typing import Optional
from config import *
from model import *
from audio_preprocessing import *

class MicrophoneCalibrator:
    def __init__(self, 
                 sample_rate: int = 16000, 
                 channels: int = 1, 
                 device: Optional[int] = None):
        self.sample_rate = sample_rate
        self.channels = channels
        self.device = device
        self.gain = 1.0
        self.factory_sound_level = None
        self.silence_level = None

    def record_segment(self, duration: float = 3.0):
        print(f"[Calib] {duration}초간 마이크 녹음 중...")
        audio = sd.rec(int(duration * self.sample_rate), samplerate=self.sample_rate, channels=self.channels, dtype='float32', device=self.device)
        sd.wait()
        return audio.flatten()

    def calibrate_factory_sound(self, yamnet_model, lstm_model, min_level=0.01, max_gain=10.0, duration=3.0):
        """
        yamnet_model: tfhub loaded model
        lstm_model: keras loaded model
        """
        gain = 1.0
        while gain <= max_gain:
            print(f"[Calib] 게인 {gain:.2f}로 공장 소리 녹음 시도...")
            audio = self.record_segment(duration)
            
            #오디오 정규화를 거치고 max_val과 rms 값을 넘김
            audio, preprocessd_info = preprocess_audio(audio, SAMPLE_RATE)
            rms, max_val = get_audio_volume(audio)

            predicted_class, max_prob, frame_predictions , _= predict_risk(audio, yamnet_model, lstm_model)

            print("=== 오디오 전처리 정보 ===")
            for k, v in preprocessd_info.items():
                if isinstance(v, float) or isinstance(v, np.floating):
                    print(f"{k:>16}: {v:.6f}")
                else:
                    print(f"{k:>16}: {v}")
            
            print("판단 결과")
            for item in frame_predictions:
                print(f"{item:.2f}", end = ' ')
            
            if predicted_class == 1:
                print(f"[Calib] 공장 소리 감지! RMS: {rms:.4f}, AI 예측 : {predicted_class}, 확률 : {max_prob}")
                self.gain = gain
                self.factory_sound_level = rms
                return True
            else:
                print(f"[Calib] 공장 소리 아님. 게인 증가.")
                gain += 0.5
        print("[Calib] 최대 게인 도달. 공장 소리 감지 실패.")
        return False

    def calibrate_silence(self, duration=3.0):
        print(f"[Calib] 무음(배경) 녹음 중...")
        audio = self.record_segment(duration) * self.gain
        rms = float(np.sqrt(np.mean(audio**2)))
        max_val = float(np.max(np.abs(audio)))
        self.silence_level = max_val
        print(f"[Calib] 무음 RMS: {rms:.4f}, 무음 Max: {max_val:.4f}")
        return max_val

    def save_calibration(self, path: str):
        # numpy float32 타입을 float로 변환
        def to_builtin(val):
            if hasattr(val, 'item'):
                return val.item()
            return float(val) if isinstance(val, (np.floating,)) else val
        data = {
            'gain': to_builtin(self.gain),
            'factory_sound_level': to_builtin(self.factory_sound_level),
            'silence_level': to_builtin(self.silence_level)
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        print(f"[Calib] 캘리브레이션 결과 저장: {path}")



if __name__ == "__main__":
    # 변수로 직접 경로와 설정 지정
    output_path = CALIBRATION_RESULT_PATH  # 원하는 경로와 파일명으로 수정
    sample_rate = SAMPLE_RATE
    channels = 1
    device = None

    calibrator = MicrophoneCalibrator(sample_rate=sample_rate, channels=channels, device=device)

    # YAMNet 및 LSTM 모델 로드
    import tensorflow_hub as hub
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    import os

    print("[Calib] YAMNet 모델 로딩 중...")
    YAMNET_MODEL_HANDLE = 'https://tfhub.dev/google/yamnet/1'
    yamnet_model = hub.load(YAMNET_MODEL_HANDLE)
    print("[Calib] YAMNet 모델 로딩 완료!")

    try:
        print(f"[Calib] LSTM 모델 로딩 중: {DANGET_DETECT_MODEL_PATH}")
        lstm_model = load_model(DANGET_DETECT_MODEL_PATH)
        print("[Calib] LSTM 모델 로딩 완료!")

        print("[Calib] 공장 소리 캘리브레이션을 시작합니다. (공장 소리를 들려주세요)")
        calibrator.calibrate_factory_sound(yamnet_model, lstm_model)

        print("5초 후 무음 켈리브레이션 시작")
        time.sleep(5)
        print("[Calib] 무음 캘리브레이션을 시작합니다. (조용히 해주세요)")
        calibrator.calibrate_silence()
        calibrator.save_calibration(output_path)        
    except FileNotFoundError:
        print(f"{DANGET_DETECT_MODEL_PATH} 경로에 모델 파일이 존재하지 않습니다.")
        


