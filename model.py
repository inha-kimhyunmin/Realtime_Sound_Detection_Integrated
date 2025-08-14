import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.models import load_model
import os

# YAMNet 임베딩 추출 함수
def get_yamnet_embedding(audio, yamnet_model):
    """
    audio: np.ndarray (float32, 1D)
    yamnet_model: tfhub loaded model
    return: np.ndarray (frames, 1024)
    """
    waveform = tf.convert_to_tensor(audio, dtype=tf.float32)
    waveform = tf.squeeze(waveform)
    yamnet_fn = yamnet_model.signatures['serving_default']
    yamnet_output = yamnet_fn(waveform=waveform)
    embeddings = yamnet_output['output_1'].numpy()  # (frames, 1024)
    return embeddings

# LSTM 모델 불러오기 함수
def load_lstm_model(model_path):
    
    if os.path.exists(model_path):
        model = load_model(model_path)
        return model
    else:
        raise FileNotFoundError(f"LSTM 모델 파일이 존재하지 않습니다: {model_path}")

# 위험 판단 함수
def predict_risk(audio, yamnet_model, lstm_model):
    """
    audio: np.ndarray (float32, 1D)
    yamnet_model: tfhub loaded model
    lstm_model: keras loaded model
    return: (predicted_class, pred_probs)
    """
    embeddings = get_yamnet_embedding(audio, yamnet_model)
    # LSTM 모델 입력 형태에 따라 차원 조정
    if len(lstm_model.input_shape) == 3:  # LSTM 기반 모델 (batch, time_steps, features)
        # 모델 입력에 맞게 패딩/자르기
        target_length = lstm_model.input_shape[1]  # 모델의 time_steps 차원
        current_length = embeddings.shape[0]
        
        if current_length < target_length:
            # 패딩
            pad_length = target_length - current_length
            embeddings = np.pad(embeddings, ((0, pad_length), (0, 0)), mode='constant')
            print(f"📏 임베딩 패딩: {current_length} → {target_length} 프레임")
        elif current_length > target_length:
            # 자르기
            embeddings = embeddings[:target_length]
            print(f"📏 임베딩 자르기: {current_length} → {target_length} 프레임")
        
        embeddings_input = np.expand_dims(embeddings, axis=0)  # (1, time_steps, 1024)
        print(f"📏 LSTM 모델용 임베딩: {embeddings_input.shape}")
        
    elif len(lstm_model.input_shape) == 2:
        print("Dense 모델 사용")
        # Dense 레이어 기반 모델 (batch, features)
        embeddings_avg = np.mean(embeddings, axis=0)
        embeddings_input = np.expand_dims(embeddings_avg, axis=0)
    else:
        raise ValueError("지원하지 않는 LSTM 모델 입력 형태입니다.")
    
    preds = lstm_model.predict(embeddings_input, verbose=0)
    # 출력 형태에 따라 처리
    if len(preds.shape) == 3:  # LSTM 출력: (batch, time_steps, num_classes)
        preds = preds[0]  # (time_steps, num_classes)
        
        # 각 클래스의 최대 확률과 위치 찾기
        for i in range(len(preds)):
            print(f"{i+1}번 프레임", round(preds[i][0],2), round(preds[i][1],2), round(preds[i][2],2), round(preds[i][3],2), round(preds[i][4],2))

        max_probs = np.max(preds, axis=0)  # 각 클래스별 최대 확률
        overall_max_prob = np.max(max_probs)
        predicted_class = np.argmax(max_probs)
        
        # 프레임별 예측에서 가장 높은 확률을 가진 프레임 찾기
        max_frame_idx = np.argmax(np.max(preds, axis=1))
        frame_predictions = preds[max_frame_idx]  # 해당 프레임의 클래스별 확률
        
    elif len(preds.shape) == 2:  # Dense 출력: (batch, num_classes)
        preds = preds[0]  # (num_classes,)
        
        #print(f"예측 확률:", round(preds[0],2), round(preds[1],2), round(preds[2],2), round(preds[3],2), round(preds[4],2))
        
        overall_max_prob = np.max(preds)
        predicted_class = np.argmax(preds)
        frame_predictions = preds
        max_frame_idx = 0  # Dense 모델은 단일 예측
    else:
        raise ValueError(f"지원하지 않는 모델 출력 형태: {preds.shape}")
    
    return predicted_class, overall_max_prob, frame_predictions, max_frame_idx