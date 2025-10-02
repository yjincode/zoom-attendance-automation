"""
메모리 효율적인 얼굴 감지 모듈
MediaPipe Face Detection을 사용하여 고성능 얼굴 탐지 제공
TensorFlow 의존성 없이 안정적인 Windows 실행 보장
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import logging
import gc
import time
from datetime import datetime, timedelta
import threading
import os
import mediapipe as mp

class FaceDetector:
    """
    MediaPipe 기반 얼굴 탐지 클래스
    Google MediaPipe를 사용하여 고성능 얼굴 감지 제공
    TensorFlow 의존성 없이 Windows에서 안정적으로 실행
    """
    
    def __init__(self, min_detection_confidence=0.7):
        """
        얼굴 탐지기 초기화
        
        Args:
            min_detection_confidence (float): 최소 탐지 신뢰도 (0.0~1.0)
        """
        self.min_detection_confidence = min_detection_confidence
        self.logger = logging.getLogger(__name__)
        
        # MediaPipe Face Detection 모델
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = None
        self.is_model_loaded = False
        self.last_detection_time = None
        self.detection_active = False
        self.detection_thread = None
        self.detection_lock = threading.Lock()
        
        # 탐지 스케줄 설정
        self.detection_interval = 60  # 1분마다
        self.detection_duration = 15  # 15초간 활성화
        
        # 초기화 시 모델 로드
        self._load_model()
        
        self.logger.info("MediaPipe 기반 얼굴 탐지기 초기화 완료")
    
    def _load_model(self):
        """
        MediaPipe Face Detection 모델 로드
        """
        try:
            if not self.is_model_loaded:
                self.logger.info("MediaPipe Face Detection 모델 로딩 중...")
                
                # MediaPipe FaceDetection 초기화
                self.face_detection = self.mp_face_detection.FaceDetection(
                    model_selection=0,  # 0: 가까운 거리, 1: 먼 거리
                    min_detection_confidence=self.min_detection_confidence
                )
                
                self.is_model_loaded = True
                self.logger.info("MediaPipe Face Detection 모델 로드 완료")
                
        except Exception as e:
            self.logger.error(f"MediaPipe 모델 로드 실패: {e}")
            self.is_model_loaded = False
    
    def _unload_model(self):
        """
        MediaPipe 모델 언로드 (메모리 절약)
        """
        try:
            if self.is_model_loaded:
                if self.face_detection:
                    self.face_detection.close()
                self.face_detection = None
                self.is_model_loaded = False
                
                # 가비지 컬렉션 강제 실행
                gc.collect()
                
                self.logger.info("MediaPipe 모델 언로드 완료 - 메모리 절약")
                
        except Exception as e:
            self.logger.error(f"모델 언로드 중 오류: {e}")
    
    def is_detection_time(self) -> bool:
        """
        현재 시간이 얼굴 감지 시간인지 확인
        교시의 35~50분 사이에만 True 반환
        
        Returns:
            bool: 감지 시간 여부
        """
        now = datetime.now()
        current_minute = now.minute
        
        # 교시별 35~50분 확인
        if 35 <= current_minute <= 50:
            return True
        
        return False
    
    def should_activate_detection(self) -> bool:
        """
        탐지를 활성화해야 하는지 확인
        1분마다 15초간 활성화
        
        Returns:
            bool: 활성화 여부
        """
        if not self.is_detection_time():
            return False
        
        now = datetime.now()
        
        # 처음 실행이거나 마지막 탐지로부터 1분 이상 경과
        if (self.last_detection_time is None or 
            now - self.last_detection_time >= timedelta(seconds=self.detection_interval)):
            return True
        
        # 현재 탐지 기간 중인지 확인 (15초간)
        if (self.last_detection_time and 
            now - self.last_detection_time <= timedelta(seconds=self.detection_duration)):
            return True
        
        return False
    
    def start_detection_cycle(self):
        """
        탐지 사이클 시작 (백그라운드 스레드)
        """
        if self.should_activate_detection():
            with self.detection_lock:
                if not self.detection_active:
                    self.detection_active = True
                    self.last_detection_time = datetime.now()
                    
                    # 모델 로드
                    self._load_model()
                    
                    # 15초 후 자동 언로드 스케줄링
                    if self.detection_thread:
                        self.detection_thread.cancel()
                    
                    self.detection_thread = threading.Timer(
                        self.detection_duration, 
                        self._end_detection_cycle
                    )
                    self.detection_thread.start()
                    
                    self.logger.info("얼굴 감지 활성화 (15초간)")
    
    def _end_detection_cycle(self):
        """
        탐지 사이클 종료
        """
        with self.detection_lock:
            if self.detection_active:
                self.detection_active = False
                self._unload_model()
                self.logger.info("얼굴 감지 비활성화 - 메모리 절약 모드")
        
    def detect_faces(self, image: np.ndarray, force_detection: bool = False) -> List[dict]:
        """
        이미지에서 얼굴을 탐지
        메모리 효율성을 위해 탐지 시간에만 작동
        
        Args:
            image (np.ndarray): BGR 형식의 이미지
            force_detection (bool): 강제 탐지 모드 (테스트용)
            
        Returns:
            List[dict]: 탐지된 얼굴 정보 리스트
                       각 dict는 'box', 'confidence', 'keypoints' 키를 포함
        """
        try:
            # 강제 탐지 모드가 아니고 탐지 시간이 아니면 빈 결과 반환
            if not force_detection and not self.should_activate_detection():
                return []
            
            # 강제 탐지 모드일 때는 즉시 모델 로드
            if force_detection:
                if not self.is_model_loaded:
                    self._load_model()
            else:
                # 탐지 사이클 시작 (필요시 모델 로드)
                self.start_detection_cycle()
            
            # 모델이 로드되지 않았으면 빈 결과 반환
            if not self.is_model_loaded or self.face_detection is None:
                return []
            
            with self.detection_lock:
                if not self.detection_active and not force_detection:
                    return []
                
                # BGR to RGB 변환 (MediaPipe는 RGB 형식을 요구)
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # 얼굴 탐지 수행
                results = self.face_detection.process(rgb_image)
                faces = []
                
                if results.detections:
                    h, w, _ = image.shape
                    for detection in results.detections:
                        # 신뢰도 추출
                        confidence = detection.score[0]
                        
                        # 바운딩 박스 추출
                        bbox = detection.location_data.relative_bounding_box
                        x = int(bbox.xmin * w)
                        y = int(bbox.ymin * h)
                        width = int(bbox.width * w)
                        height = int(bbox.height * h)
                        
                        # 키포인트 추출
                        keypoints = {}
                        for idx, keypoint in enumerate(detection.location_data.relative_keypoints):
                            keypoints[f'keypoint_{idx}'] = (int(keypoint.x * w), int(keypoint.y * h))
                        
                        faces.append({
                            'box': [x, y, width, height],
                            'confidence': confidence,
                            'keypoints': keypoints
                        })
                
                self.logger.debug(f"탐지된 얼굴 수: {len(faces)}")
                return faces
            
        except Exception as e:
            self.logger.error(f"얼굴 탐지 중 오류 발생: {e}")
            return []
    
    def has_faces(self, image: np.ndarray, confidence_threshold=0.9, force_detection: bool = False) -> bool:
        """
        이미지에 얼굴이 있는지 확인
        메모리 효율성을 위해 탐지 시간에만 실제 검사 수행
        
        Args:
            image (np.ndarray): BGR 형식의 이미지
            confidence_threshold (float): 얼굴 탐지 신뢰도 임계값
            force_detection (bool): 강제 탐지 모드
            
        Returns:
            bool: 얼굴이 탐지되면 True, 아니면 False
        """
        # 강제 탐지 모드가 아니고 탐지 시간이 아니면 False 반환 (메모리 절약)
        if not force_detection and not self.should_activate_detection():
            return False
        
        faces = self.detect_faces(image, force_detection=force_detection)
        
        # 신뢰도가 임계값 이상인 얼굴이 하나 이상 있는지 확인
        valid_faces = [face for face in faces if face['confidence'] >= confidence_threshold]
        
        return len(valid_faces) > 0
    
    def force_detection(self, image: np.ndarray) -> List[dict]:
        """
        시간 제약 없이 강제로 얼굴 탐지 (테스트용)
        
        Args:
            image (np.ndarray): BGR 형식의 이미지
            
        Returns:
            List[dict]: 탐지된 얼굴 정보 리스트
        """
        return self.detect_faces(image, force_detection=True)
    
    def get_memory_status(self) -> dict:
        """
        현재 메모리 상태 및 탐지 상태 반환
        
        Returns:
            dict: 상태 정보
        """
        return {
            'model_loaded': self.is_model_loaded,
            'detection_active': self.detection_active,
            'is_detection_time': self.is_detection_time(),
            'should_activate': self.should_activate_detection(),
            'last_detection': self.last_detection_time.isoformat() if self.last_detection_time else None
        }
    
    def cleanup(self):
        """
        리소스 정리 (프로그램 종료 시 호출)
        """
        try:
            # 탐지 스레드 정리
            if self.detection_thread:
                self.detection_thread.cancel()
            
            # 모델 언로드
            self._unload_model()
            
            self.logger.info("얼굴 탐지기 리소스 정리 완료")
            
        except Exception as e:
            self.logger.error(f"리소스 정리 중 오류: {e}")
    
    def draw_faces(self, image: np.ndarray, save_path: Optional[str] = None, force: bool = False) -> np.ndarray:
        """
        탐지된 얼굴에 박스를 그려서 시각화
        
        Args:
            image (np.ndarray): BGR 형식의 이미지
            save_path (str, optional): 저장할 경로
            force (bool): 강제 탐지 여부 (테스트용)
            
        Returns:
            np.ndarray: 얼굴 박스가 그려진 이미지
        """
        if force:
            faces = self.force_detection(image)
        else:
            faces = self.detect_faces(image)
        
        result_image = image.copy()
        
        # 메모리 상태 정보 추가
        status = self.get_memory_status()
        status_text = f"MediaPipe: {'Loaded' if status['model_loaded'] else 'Unloaded'} | " \
                     f"Active: {'Yes' if status['detection_active'] else 'No'} | " \
                     f"Time: {'Yes' if status['is_detection_time'] else 'No'}"
        
        cv2.putText(result_image, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        for face in faces:
            x, y, w, h = face['box']
            confidence = face['confidence']
            
            # 얼굴 영역에 사각형 그리기
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # 신뢰도 텍스트 추가
            text = f"Face: {confidence:.2f}"
            cv2.putText(result_image, text, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # 키포인트 그리기 (눈, 코, 입)
            if 'keypoints' in face:
                for point_name, (px, py) in face['keypoints'].items():
                    cv2.circle(result_image, (px, py), 3, (255, 0, 0), -1)
        
        if save_path:
            cv2.imwrite(save_path, result_image)
            
        return result_image

def calculate_image_sharpness(image: np.ndarray) -> float:
    """
    이미지의 선명도를 계산하는 함수
    Laplacian 연산자의 분산을 사용하여 선명도 측정
    
    Args:
        image (np.ndarray): BGR 형식의 이미지
        
    Returns:
        float: 선명도 점수 (높을수록 선명)
    """
    # 그레이스케일 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Laplacian 연산자 적용
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    
    # 분산 계산 (선명도 지표)
    sharpness = laplacian.var()
    
    return sharpness

# 테스트 코드
if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    
    # 메모리 효율적인 얼굴 탐지기 초기화
    detector = FaceDetector()
    
    print("MediaPipe 얼굴 탐지기 테스트")
    print("=" * 50)
    
    # 현재 상태 확인
    status = detector.get_memory_status()
    print(f"초기 상태: {status}")
    
    # 웹캠으로 테스트 (선택사항)
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("\n웹캠 테스트 시작...")
            print("- 일반 모드: 35~50분에만 탐지 활성화")
            print("- 강제 모드: 'f' 키로 강제 탐지")
            print("- 종료: ESC 키")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 키 입력 확인
                key = cv2.waitKey(1) & 0xFF
                
                if key == 27:  # ESC
                    break
                elif key == ord('f'):  # 강제 탐지
                    result = detector.draw_faces(frame, force=True)
                    print("강제 탐지 실행")
                else:
                    result = detector.draw_faces(frame)
                
                # 상태 정보 표시
                current_time = datetime.now().strftime("%H:%M:%S")
                cv2.putText(result, f"Time: {current_time}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.imshow("MediaPipe Face Detection", result)
            
            cap.release()
            cv2.destroyAllWindows()
        else:
            print("웹캠을 사용할 수 없습니다.")
    except Exception as e:
        print(f"웹캠 테스트 오류: {e}")
    
    # 리소스 정리
    detector.cleanup()
    print("MediaPipe 얼굴 탐지 모듈 테스트 완료")