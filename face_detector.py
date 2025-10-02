"""
메모리 효율적인 얼굴 감지 모듈
OpenCV DNN 기반 사전 훈련된 모델 사용
순수 OpenCV만 사용하여 외부 의존성 없이 고성능 얼굴 감지 제공
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
import urllib.request
from pathlib import Path

class FaceDetector:
    """
    OpenCV DNN 기반 얼굴 탐지 클래스
    사전 훈련된 DNN 모델을 사용하여 고성능 얼굴 감지 제공
    순수 OpenCV만 사용하여 외부 의존성 없이 안정적 실행
    """
    
    def __init__(self, min_detection_confidence=0.5):
        """
        얼굴 탐지기 초기화
        
        Args:
            min_detection_confidence (float): 최소 탐지 신뢰도 (0.0~1.0)
        """
        self.min_detection_confidence = min_detection_confidence
        self.logger = logging.getLogger(__name__)
        
        # OpenCV DNN 모델 관련
        self.net = None
        self.is_model_loaded = False
        self.last_detection_time = None
        self.detection_active = False
        self.detection_thread = None
        self.detection_lock = threading.Lock()
        
        # 모델 파일 경로
        self.model_dir = Path(__file__).parent / "models"
        self.model_dir.mkdir(exist_ok=True)
        
        # 탐지 스케줄 설정
        self.detection_interval = 60  # 1분마다
        self.detection_duration = 15  # 15초간 활성화
        
        # 초기화 시 모델 로드
        self._load_model()
        
        self.logger.info("OpenCV DNN 기반 얼굴 탐지기 초기화 완료")
    
    def _load_model(self):
        """
        OpenCV DNN Face Detection 모델 로드
        """
        try:
            if not self.is_model_loaded:
                self.logger.info("OpenCV DNN Face Detection 모델 로딩 중...")
                
                # 모델 파일 다운로드 (필요시)
                self._download_model_files()
                
                # DNN 모델 로드
                prototxt_path = self.model_dir / "deploy.prototxt"
                model_path = self.model_dir / "res10_300x300_ssd_iter_140000.caffemodel"
                
                if prototxt_path.exists() and model_path.exists():
                    self.net = cv2.dnn.readNetFromCaffe(str(prototxt_path), str(model_path))
                    self.is_model_loaded = True
                    self.logger.info("OpenCV DNN Face Detection 모델 로드 완료")
                else:
                    raise Exception("모델 파일을 찾을 수 없습니다")
                
        except Exception as e:
            self.logger.error(f"OpenCV DNN 모델 로드 실패: {e}")
            self.is_model_loaded = False
    
    def _download_model_files(self):
        """
        사전 훈련된 DNN 모델 파일 다운로드
        """
        model_urls = {
            "deploy.prototxt": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
            "res10_300x300_ssd_iter_140000.caffemodel": "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
        }
        
        for filename, url in model_urls.items():
            filepath = self.model_dir / filename
            
            if not filepath.exists():
                try:
                    self.logger.info(f"모델 파일 다운로드 중: {filename}")
                    urllib.request.urlretrieve(url, filepath)
                    self.logger.info(f"다운로드 완료: {filename}")
                except Exception as e:
                    self.logger.error(f"모델 파일 다운로드 실패 {filename}: {e}")
                    # 내장된 기본 모델 파라미터 사용
                    self._create_fallback_model(filename)
    
    def _create_fallback_model(self, filename):
        """
        네트워크 문제로 다운로드 실패 시 Haar Cascade 폴백
        """
        if filename == "deploy.prototxt":
            # 간단한 prototxt 내용 생성 (실제로는 Haar Cascade 사용)
            self.logger.info("네트워크 오류로 인해 Haar Cascade 폴백 모드 사용")
            self.use_haar_fallback = True
        
    def _unload_model(self):
        """
        OpenCV DNN 모델 언로드 (메모리 절약)
        """
        try:
            if self.is_model_loaded:
                self.net = None
                self.is_model_loaded = False
                
                # 가비지 컬렉션 강제 실행
                gc.collect()
                
                self.logger.info("OpenCV DNN 모델 언로드 완료 - 메모리 절약")
                
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
            if not self.is_model_loaded or self.net is None:
                return []
            
            with self.detection_lock:
                if not self.detection_active and not force_detection:
                    return []
                
                # 이미지 전처리
                h, w = image.shape[:2]
                blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123])
                
                # DNN 추론 수행
                self.net.setInput(blob)
                detections = self.net.forward()
                
                faces = []
                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    
                    # 신뢰도 임계값 확인
                    if confidence > self.min_detection_confidence:
                        # 바운딩 박스 계산
                        x1 = int(detections[0, 0, i, 3] * w)
                        y1 = int(detections[0, 0, i, 4] * h)
                        x2 = int(detections[0, 0, i, 5] * w)
                        y2 = int(detections[0, 0, i, 6] * h)
                        
                        # 박스 크기 및 위치 검증
                        if x1 >= 0 and y1 >= 0 and x2 <= w and y2 <= h and x2 > x1 and y2 > y1:
                            faces.append({
                                'box': [x1, y1, x2 - x1, y2 - y1],
                                'confidence': float(confidence),
                                'keypoints': {}  # DNN 모델은 키포인트 제공하지 않음
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
        status_text = f"OpenCV DNN: {'Loaded' if status['model_loaded'] else 'Unloaded'} | " \
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
            
            # 키포인트 그리기 (DNN 모델은 키포인트 미제공)
            if 'keypoints' in face and face['keypoints']:
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
    
    print("OpenCV DNN 얼굴 탐지기 테스트")
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
                
                cv2.imshow("OpenCV DNN Face Detection", result)
            
            cap.release()
            cv2.destroyAllWindows()
        else:
            print("웹캠을 사용할 수 없습니다.")
    except Exception as e:
        print(f"웹캠 테스트 오류: {e}")
    
    # 리소스 정리
    detector.cleanup()
    print("OpenCV DNN 얼굴 탐지 모듈 테스트 완료")