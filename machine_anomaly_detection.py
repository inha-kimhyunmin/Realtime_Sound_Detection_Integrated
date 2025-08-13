# !pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
# !pip install onnxruntime==1.22.1
# !pip install numpy==2.0.2

# ==============================================================================
# 필요한 라이브러리 임포트
# ==============================================================================
import os
import time
import json
import torch
import numpy as np
import onnxruntime
import torch.nn.functional as F
import torchaudio.transforms as T
from config import PART_CONFIGS

# ==============================================================================
# 헬퍼 함수들
# ==============================================================================
def compute_spectral_std(spec, dim=-1, keepdim=True):
   return spec.std(dim=dim, keepdim=keepdim)

def compute_spectral_kurtosis(spec, dim=-1, eps=1e-6):
   mean = spec.mean(dim=dim, keepdim=True)
   std = spec.std(dim=dim, keepdim=True) + eps
   standardized = (spec - mean) / std
   return (standardized ** 4).mean(dim=dim, keepdim=True)

def compute_spectral_skewness(spec, dim=-1, eps=1e-6):
   mean = spec.mean(dim=dim, keepdim=True)
   std = spec.std(dim=dim, keepdim=True) + eps
   standardized = (spec - mean) / std
   return (standardized ** 3).mean(dim=dim, keepdim=True)

def compute_delta_std(spec, dim=-1, keepdim=True):
   delta = F.pad(spec[..., 1:] - spec[..., :-1], (1, 0))
   return delta.std(dim=dim, keepdim=keepdim)

def normalize_db(waveform, target_db=-12.0):
   rms = waveform.pow(2).mean().sqrt()
   scalar = (10 ** (target_db / 20)) / (rms + 1e-8)
   return waveform * scalar

def convert_to_melspectrogram(waveform, sample_rate, n_mels=224, n_fft=2048, hop_length=512, target_width=224):
   mel_spectrogram_transform = T.MelSpectrogram(
       sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
   )
   mel_spec = mel_spectrogram_transform(waveform)
   if mel_spec.shape[-1] < target_width:
       pad_amount = target_width - mel_spec.shape[-1]
       mel_spec = F.pad(mel_spec, (0, pad_amount))
   elif mel_spec.shape[-1] > target_width:
       mel_spec = mel_spec[..., :target_width]
   return mel_spec

def preprocess_spectrogram_for_inference(spectrogram, part_name):
   if part_name.lower() == 'gearbox':
       feature_channel = compute_spectral_std(spectrogram).expand_as(spectrogram)
   elif part_name.lower() == 'bearing':
       feature_channel = compute_spectral_kurtosis(spectrogram).expand_as(spectrogram)
   elif part_name.lower() == 'fan':
       feature_channel = compute_spectral_skewness(spectrogram).expand_as(spectrogram)
   elif part_name.lower() == 'slider':
       feature_channel = compute_delta_std(spectrogram).expand_as(spectrogram)
   elif part_name.lower() == 'pump':
       feature_channel = compute_spectral_skewness(spectrogram).expand_as(spectrogram)
   else:
       feature_channel = torch.zeros_like(spectrogram)
   
   eps = 1e-6
   x_min, x_max = spectrogram.min(), spectrogram.max()
   x_normalized = (spectrogram - x_min) / (x_max - x_min + eps)
   f_min, f_max = feature_channel.min(), feature_channel.max()
   feature_normalized = (feature_channel - f_min) / (f_max - f_min + eps)
   model_input = torch.cat([x_normalized, feature_normalized], dim=0)
   return model_input.unsqueeze(0)

def sigmoid_recalibration(raw_prob, threshold, steepness=20):
   x = (raw_prob - threshold) * steepness
   calibrated = 1 / (1 + np.exp(-x))
   return calibrated

# ==============================================================================
# 메인 추론 함수
# ==============================================================================
def run_inference(segment_np, part_name):
   """
   오디오 세그먼트에서 특정 부품의 이상을 탐지
   
   Args:
       segment_np: numpy array (160000,)
       part_name: 부품명 ('bearing', 'fan', 'gearbox', 'pump', 'slider')
   
   Returns:
       결과 딕셔너리
   """
   # 부품 설정 가져오기
   config = PART_CONFIGS[part_name]
   model_path = config['model_path']
   threshold = config['threshold']
   
   # ONNX 모델 로드
   session = onnxruntime.InferenceSession(model_path)
   input_name = session.get_inputs()[0].name
   
   # 전처리
   waveform = torch.tensor(segment_np, dtype=torch.float32).unsqueeze(0)
   waveform_normalized = normalize_db(waveform)
   spectrogram = convert_to_melspectrogram(waveform_normalized, 16000)
   model_input = preprocess_spectrogram_for_inference(spectrogram, part_name)
   
   # 추론
   output = session.run(None, {input_name: model_input.cpu().numpy()})[0]
   logit = output.item()
   raw_prob = 1 / (1 + np.exp(-logit))
   calibrated_prob = sigmoid_recalibration(raw_prob, threshold)
   
   # 결과 반환
   return {
       "device_name": f"Factory-1_{part_name.upper()}",
       "result": bool(calibrated_prob >= 0.5), # true 이상 , False 정상
       "probability": float(format(calibrated_prob, '.3f')), # 이상일 확률
       "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
   }


def run_inference_all(segment_np, part_names):
    """
    특정 부품 이상 탐지를 한번에 실행
    part_names : 부품 리스트
    """
    result = []
    for part_name in part_names:
        result.append(run_inference(segment_np, part_name))

    return result



# ==============================================================================
# 사용 예시
# ==============================================================================
if __name__ == "__main__":
   # 10초 오디오 데이터
   audio_segment = np.random.randn(160000).astype(np.float32) * 0.1
   
   # bearing 검사
   result = run_inference(audio_segment, 'bearing')
   print(json.dumps(result, indent=2, ensure_ascii=False))
   
   # fan 검사
   result = run_inference(audio_segment, 'fan')
   print(json.dumps(result, indent=2, ensure_ascii=False))