import sounddevice as sd
import numpy as np
import json
import threading
import tensorflow_hub as hub
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import json

from typing import Optional
from config import *
from model import predict_risk
import queue
from audio_preprocessing import *
from Realtime_recording import RealtimeSegmentRecorder
from machine_anomaly_detection import *


if __name__ == "__main__":
    # 변수로 직접 경로와 설정 지정
    sample_rate = SAMPLE_RATE
    channels = 1
    device = None

    print("YAMNet 모델 로딩 중...")
    YAMNET_MODEL_HANDLE = 'https://tfhub.dev/google/yamnet/1'
    yamnet_model = hub.load(YAMNET_MODEL_HANDLE)
    print("YAMNet 모델 로딩 완료!")

    lstm_model = load_model(DANGET_DETECT_MODEL_PATH)
    print("LSTM 모델 로딩 완료!")
    # 녹음기 생성 및 시작
    recorder = RealtimeSegmentRecorder(sample_rate=sample_rate, channels=channels, segment_duration=SEGMENT_DURATION)
    audio_segments = []
    try:
        recorder.start()
        print("[Main] Recording... (Ctrl+C to stop)")
        while True:
            try:
                audio = recorder.get_segment(timeout=1)

                #일반 전처리
                audio, preprocess_info = preprocess_audio(audio, SAMPLE_RATE)

                #캘리브레이션 데이터
                with open("calibration_result.json", "r") as f:
                    calibration_data = json.load(f)
                gain = calibration_data["gain"]
                factory_sound_level = calibration_data["factory_sound_level"]
                silence_level = calibration_data["silence_level"]
                
                #오디오 게인 처리
                audio = audio * gain

                #오디오 음량 파악
                rms, max_val = get_audio_volume(audio)

                #10초 세그먼트 저장용
                audio_segments.append(audio)
                
                predicted_classes = []
                if USE_MUTE_BY_FACTORY_SOUND:
                    if rms >= silence_level and rms >= factory_sound_level * FACTORY_SOUND_RATIO:
                        predicted_class, overall_max_prob, frame_predictions, max_frame_idx = predict_risk(audio, yamnet_model, lstm_model)
                        print(f"[Result] Predicted class: {predicted_class}")
                        print(f"         Overall max prob: {overall_max_prob:.4f}")
                        print(f"         Frame predictions: {frame_predictions}")
                        print(f"         Max prob frame idx: {max_frame_idx}")
                    else:
                        print("임계값보다 소리가 작아 무음으로 처리")
                        predicted_class = 0
                else:
                    if rms >= silence_level:
                        predicted_class, overall_max_prob, frame_predictions, max_frame_idx = predict_risk(audio, yamnet_model, lstm_model)
                        print(f"[Result] Predicted class: {predicted_class}")
                        print(f"         Overall max prob: {overall_max_prob:.4f}")
                        print(f"         Frame predictions: {frame_predictions}")
                        print(f"         Max prob frame idx: {max_frame_idx}")
                    else:
                        print("임계값보다 소리가 작아 무음으로 처리")
                        predicted_class = 0

                predicted_classes.append(predicted_class)
                if len(audio_segments) == MACHINE_ANOMALY_INPUT_DURATION / SEGMENT_DURATION:  # 원하는 길이(2개)가 되면
                    combined_audio = np.concatenate(audio_segments)
                    if not any(x != 1 for x in predicted_classes) and len(combined_audio) == int(MACHINE_ANOMALY_INPUT_DURATION * sample_rate):
                        machine_sound_result = run_inference_all(combined_audio, DETECT_PARTS)
                        print(machine_sound_result)
                    audio_segments = []  # 초기화 또는 슬라이딩 윈도우로 관리

            except queue.Empty:
                continue
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        recorder.stop()


