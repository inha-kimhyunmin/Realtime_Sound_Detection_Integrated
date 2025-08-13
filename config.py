
# 기본 오디오 설정
SAMPLE_RATE = 16000
SEGMENT_DURATION = 5  # 실시간 처리 단위 (초)
DANGET_DETECT_MODEL_PATH = 'model/Danger_detect/yamnet_lstm_model_20250812_173732.h5'
CALIBRATION_RESULT_PATH = 'calibration_result.json'

DANGER_DETECT_INPUT_DURATION = 5
MACHINE_ANOMALY_INPUT_DURATION = 10

PART_CONFIGS = {
   'bearing': {
       'model_path': 'model/Machine_anomaly_detect/bearing.onnx',
       'threshold': 0.090
   },
   'fan': {
       'model_path': 'model/Machine_anomaly_detect/fan.onnx',
       'threshold': 0.505
   },
   'gearbox': {
       'model_path': 'model/Machine_anomaly_detect/gearbox.onnx',
       'threshold': 0.245
   },
   'pump': {
       'model_path': 'model/Machine_anomaly_detect/pump.onnx',
       'threshold': 0.190
   },
   'slider': {
       'model_path': 'model/Machine_anomaly_detect/slider.onnx',
       'threshold': 0.050
   }
}

DETECT_PARTS = ['bearing', 'gearbox', 'fan', 'pump', 'slider']
USE_MUTE_BY_FACTORY_SOUND = True 
FACTORY_SOUND_RATIO = 0.5