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

"""
모델 로딩 오류가 난다면 CMD에서
rmdir /s "C:/Users/dotor\AppData\Local\Temp\tfhub_modules"

"""

def danger_detect_output(predicted_class, overall_max_prob, frame_predictions):

    print("환경 소리 AI모델 판단 결과")
    for i in range(len(frame_predictions)):
        print(f"{CLASSES[i]} : {frame_predictions[i]:.2f}")
    
    if predicted_class <= FACTORY_CLASS:
        if overall_max_prob >= FACTORY_THRESHOLD:
            print(f"AI 예측 : {CLASSES[predicted_class]}, 확률 : {overall_max_prob:.2f}")
        else:
            print(f"확률 임계값 미만으로 무음 처리, AI 판단 : {CLASSES[predicted_class]}, 확률 : {overall_max_prob:.2f}")
    else:
        if overall_max_prob >= DANGER_THRESHOLD:
            print(f"AI 예측 : {CLASSES[predicted_class]}, 확률 : {overall_max_prob:.2f}")
        else:
            print(f"확률 임계값 미만으로 무음 처리 \n AI 판단 : {CLASSES[predicted_class]}, 확률 : {overall_max_prob:.2f}")
    return None

def machine_anomaly_detect_output(result):
    for dict in result:
        device_name = dict['device_name']        # 'Factory-1_BEARING'
        result = dict['result']                  # True
        probability = dict['probability']        # 0.943
        created_at = dict['created_at']          # 시간
        print(f"부품명 : {device_name}, 정상 여부 : {result}, 확률 : {probability}, 시간 : {created_at}")
    return None

if __name__ == "__main__":
    print("YAMNet 모델 로딩 중...")
    YAMNET_MODEL_HANDLE = 'https://tfhub.dev/google/yamnet/1'
    yamnet_model = hub.load(YAMNET_MODEL_HANDLE)
    print("YAMNet 모델 로딩 완료!")

    lstm_model = load_model(DANGET_DETECT_MODEL_PATH)
    print("LSTM 모델 로딩 완료!")
    # 녹음기 생성 및 시작
    recorder = RealtimeSegmentRecorder(sample_rate=SAMPLE_RATE, channels=MIC_NUM, segment_duration=SEGMENT_DURATION)
    audio_segments = []
    predicted_classes = []
    try:
        recorder.start()
        print("[Main] Recording... (Ctrl+C to stop)")
        while True:
            try:
                audio = recorder.get_segment(timeout=1)
                
                #다채널 입력 처리
                start_time = time.time()
                for mic_num in range(MIC_NUM):
                    single_audio = audio[:, mic_num]
                    #일반 전처리
                    single_audio, preprocess_info = preprocess_audio(single_audio, SAMPLE_RATE)

                    #캘리브레이션 데이터
                    with open("calibration_result.json", "r") as f:
                        calibration_data = json.load(f)
                    gain = calibration_data["gain"]
                    factory_sound_level = calibration_data["factory_sound_level"]
                    silence_level = calibration_data["silence_level"]

                    #오디오 게인 처리
                    single_audio = single_audio * gain

                    #오디오 음량 파악
                    rms, max_val = get_audio_volume(single_audio)

                    #10초 세그먼트 저장용
                    audio_segments.append(single_audio)

                    #환경 소리 모델 판단
                    print(f"RMS : {rms:.4f}, MAX : {max_val:.4f}")
                    print(f"{mic_num + 1}번 마이크 판단")
                    if USE_MUTE_BY_FACTORY_SOUND:
                        if rms >= silence_level and rms >= factory_sound_level * FACTORY_SOUND_RATIO:
                            predicted_class, overall_max_prob, frame_predictions, _ = predict_risk(single_audio, yamnet_model, lstm_model)
                            danger_detect_output(predicted_class, overall_max_prob, frame_predictions)
                        else:
                            print(f"임계값({max(factory_sound_level*FACTORY_SOUND_RATIO, silence_level):.4f})보다 소리가 작아 무음으로 처리")
                            predicted_class = 0
                    else:
                        if rms >= silence_level:
                            predicted_class, overall_max_prob, frame_predictions, max_frame_idx = predict_risk(single_audio, yamnet_model, lstm_model)
                            danger_detect_output(predicted_class, overall_max_prob, frame_predictions)
                        else:
                            print(f"임계값({silence_level:.4f})보다 소리가 작아 무음으로 처리")
                            predicted_class = 0

                    #환경 소리 모델 판단 세그먼트별 저장
                    #공장 소리가 연속 2번 들어오면 넘겨야함(다른 클래스가 감지되면 리스트 초기화)
                    if predicted_class == FACTORY_CLASS:
                        predicted_classes.append(predicted_class)
                    else:
                        predicted_classes.clear()

                    #기계 이상 진단 모델 판단
                    if len(audio_segments) == MACHINE_ANOMALY_INPUT_DURATION / SEGMENT_DURATION:  # 원하는 길이(2개)가 되면
                        combined_audio = np.concatenate(audio_segments)
                        if len(predicted_classes) >= MACHINE_ANOMALY_INPUT_DURATION / SEGMENT_DURATION and len(combined_audio) == int(MACHINE_ANOMALY_INPUT_DURATION * SAMPLE_RATE): 
                            #5초 판단 2개가 전부 공장 소리이고, 오디오 길이가 원하는 길이가 되면
                            machine_sound_result = run_inference_all(combined_audio, DETECT_PARTS[mic_num])
                            machine_anomaly_detect_output(machine_sound_result)
                            predicted_classes.clear()
                        audio_segments = []  # 초기화 또는 슬라이딩 윈도우로 관리
                end_time = time.time()
                print(f"총 판단 시간 : {(end_time - start_time):.2f}초")
                print()
            except queue.Empty:
                continue
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        recorder.stop()