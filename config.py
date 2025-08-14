
# 기본 오디오 설정
SAMPLE_RATE = 16000
SEGMENT_DURATION = 5  # 실시간 처리 단위 (초)
DANGET_DETECT_MODEL_PATH = 'model/Danger_detect/yamnet_lstm_model_20250812_173732.h5'
CALIBRATION_RESULT_PATH = 'calibration_result.json'
MIC_NUM = 2 #채널 수(시연시는 1로 조절)

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

#환경 소리 판단 클래스
CLASSES = ['Silence', 'Factory', 'Fire', 'Gas_Leak', 'Scream']
#공장 클래스 번호
FACTORY_CLASS = CLASSES.index('Factory')
#공장 소리 판단 임계 확률
FACTORY_THRESHOLD = 0.6
#위험 소리 판단 임계 확률
DANGER_THRESHOLD = 0.7
#판단할 기계 부품목록(시연시 이중 하나만 입력하면된다.) 각 채널별로 몇 개 만들것인가
DETECT_PARTS = [['bearing', 'gearbox', 'fan', 'pump', 'slider'], 
                ['bearing', 'gearbox']    
               ]

#기계 소리의 일정 비율 이하 소리는 무음처리
USE_MUTE_BY_FACTORY_SOUND = True 
#기계 소리의 몇 %보다 작으면 무음이냐?
FACTORY_SOUND_RATIO = 0.5