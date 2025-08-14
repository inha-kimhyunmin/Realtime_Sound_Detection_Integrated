import sounddevice as sd
import numpy as np
import threading
import queue
from collections import deque
from typing import Callable, Optional
from config import *

class RealtimeSegmentRecorder:
    def __init__(self, 
                 sample_rate: int = 16000, 
                 channels: int = 1, 
                 segment_duration: float = 10.0, 
                 callback: Optional[Callable] = None,
                 device: Optional[int] = None):
        self.sample_rate = sample_rate
        self.channels = channels
        self.segment_samples = int(sample_rate * segment_duration)
        self.callback = callback
        self.device = device

        self.buffer = deque(maxlen=self.segment_samples * 2)  # 오버랩 지원
        self.segment_queue = queue.Queue()
        self.running = False

    def _audio_callback(self, indata, frames, time, status):
        # indata: shape (frames, channels)
        # 프레임 단위(1D array, shape: (channels,))로 append
        for frame in indata:
            self.buffer.append(frame.copy())

    def _segment_worker(self):
        while self.running:
            if len(self.buffer) >= self.segment_samples:
                # 세그먼트 추출 (샘플, 채널)
                segment = np.array([self.buffer.popleft() for _ in range(self.segment_samples)], dtype=np.float32)
                # segment.shape == (segment_samples, channels)
                if self.callback:
                    self.callback(segment)
                else:
                    self.segment_queue.put(segment)
            else:
                # 데이터가 충분하지 않으면 잠시 대기
                threading.Event().wait(0.05)

    def start(self):
        self.running = True
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            callback=self._audio_callback,
            device=self.device,
            dtype='float32',
            blocksize=0
        )
        self.stream.start()
        self.worker_thread = threading.Thread(target=self._segment_worker, daemon=True)
        self.worker_thread.start()

    def stop(self):
        self.running = False
        self.stream.stop()
        self.stream.close()
        self.worker_thread.join(timeout=1.0)

    def get_segment(self, timeout=None):
        """콜백 대신 큐에서 세그먼트를 직접 꺼내고 싶을 때 사용"""
        return self.segment_queue.get(timeout=timeout)

# 사용 예시
if __name__ == "__main__":
    def print_segment(segment):
        print(f"Segment shape: {segment.shape}, RMS: {np.sqrt(np.mean(segment**2)):.4f}")
        print(segment, segment[0], segment[1], len(segment[0]), len(segment[1]))

    recorder = RealtimeSegmentRecorder(sample_rate=SAMPLE_RATE, channels= MIC_NUM, segment_duration=SEGMENT_DURATION, callback=print_segment)
    
    try:
        recorder.start()
        print("Recording... (Ctrl+C to stop)")
        while True:
            threading.Event().wait(1)
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        recorder.stop()