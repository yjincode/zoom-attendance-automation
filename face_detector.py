"""
메모리 효율적인 얼굴 감지 모듈
MTCNN을 사용하여 이미지에서 얼굴을 탐지하되, 메모리 사용량을 최적화
교시별 특정 시간(35~50분)에만 1분마다 15초간 모델 로드하여 리소스 절약
"""

import cv2
import numpy as np
from mtcnn import MTCNN
from typing import List, Tuple, Optional
import logging
import gc
import time
from datetime import datetime, timedelta
import threading

class FaceDetector:
    """
    메모리 효율적인 MTCNN 얼굴 탐지 클래스
    필요할 때만 모델을 로드하여 메모리 사용량 최적화
    교시별 35~50분 시간대에만 1분마다 15초간 활성화
    """
    
    def __init__(self, min_face_size=20):
        """
        얼굴 탐지기 초기화
        
        Args:
            min_face_size (int): 탐지할 최소 얼굴 크기 (픽셀)
        """
        self.min_face_size = min_face_size
        self.logger = logging.getLogger(__name__)
        
        # 메모리 효율성을 위한 변수들
        self.detector = None
        self.is_model_loaded = False
        self.last_detection_time = None
        self.detection_active = False
        self.detection_thread = None
        self.detection_lock = threading.Lock()
        
        # 탐지 스케줄 설정
        self.detection_interval = 60  # 1분마다
        self.detection_duration = 15  # 15초간 활성화
        
        self.logger.info("메모리 효율적인 얼굴 탐지기 초기화 완료")
    
    def _load_model(self):
        """
        MTCNN 모델 로드 (필요할 때만)
        """
        try:
            if not self.is_model_loaded:
                self.logger.info("MTCNN 모델 로딩 중...")
                self.detector = MTCNN(min_face_size=self.min_face_size)
                self.is_model_loaded = True
                self.logger.info("MTCNN 모델 로드 완료")
                
        except Exception as e:
            self.logger.error(f"MTCNN 모델 로드 실패: {e}")
            self.is_model_loaded = False
    
    def _unload_model(self):
        """
        MTCNN 모델 언로드 (메모리 절약)
        """
        try:
            if self.is_model_loaded:
                self.detector = None
                self.is_model_loaded = False
                
                # 가비지 컬렉션 강제 실행
                gc.collect()
                
                self.logger.info("MTCNN 모델 언로드 완료 - 메모리 절약")
                
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
            if not self.is_model_loaded or self.detector is None:
                return []
            
            with self.detection_lock:
                if not self.detection_active:
                    return []
                
                # BGR to RGB 변환 (MTCNN은 RGB 형식을 요구)
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # 얼굴 탐지 수행
                faces = self.detector.detect_faces(rgb_image)
                
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
        try:
            # 강제로 모델 로드
            if not self.is_model_loaded:
                self._load_model()
            
            if not self.is_model_loaded or self.detector is None:
                return []
            
            # BGR to RGB 변환
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 얼굴 탐지 수행
            faces = self.detector.detect_faces(rgb_image)
            
            self.logger.info(f"강제 탐지 결과: {len(faces)}개 얼굴")
            return faces
            
        except Exception as e:
            self.logger.error(f"강제 얼굴 탐지 중 오류: {e}")
            return []
    
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
        status_text = f"Model: {'Loaded' if status['model_loaded'] else 'Unloaded'} | " \
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
    
    print("🧠 메모리 효율적인 얼굴 탐지기 테스트")
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
                
                cv2.imshow("Memory Efficient Face Detection", result)
            
            cap.release()
            cv2.destroyAllWindows()
        else:
            print("웹캠을 사용할 수 없습니다.")
    except Exception as e:
        print(f"웹캠 테스트 오류: {e}")
    
    # 리소스 정리
    detector.cleanup()
    print("메모리 효율적인 얼굴 탐지 모듈 테스트 완료")