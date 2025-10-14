"""
메모리 효율적인 얼굴 감지 모듈
YuNet 기반 고정확도 얼굴 탐지 (OpenCV 4.5.4+)
순수 OpenCV만 사용하여 TensorFlow 없이 안정적 실행
Windows 호환성 우수, 얼굴 랜드마크 5개 포인트 제공
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
    YuNet 기반 고정확도 얼굴 탐지 클래스
    OpenCV 4.5.4+ 내장 YuNet 모델 사용
    TensorFlow 불필요, Windows에서 안정적, 얼굴 랜드마크 5개 포인트 제공
    """

    def __init__(self, min_detection_confidence=0.6):
        """
        YuNet 얼굴 탐지기 초기화

        Args:
            min_detection_confidence (float): 최소 탐지 신뢰도 (0.0~1.0)
        """
        self.min_detection_confidence = min_detection_confidence
        self.logger = logging.getLogger(__name__)

        # YuNet 모델 관련
        self.detector = None
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

        # YuNet 입력 크기
        self.input_size = (320, 320)

        # 초기화 시 모델 로드
        self._load_model()

        self.logger.info("YuNet 고정확도 얼굴 탐지기 초기화 완료")
    
    def _load_model(self):
        """
        YuNet ONNX 모델 로드
        """
        try:
            if not self.is_model_loaded:
                self.logger.info("YuNet 모델 로딩 중...")

                # 모델 파일 다운로드 (필요시)
                model_path = self._download_yunet_model()

                if model_path.exists():
                    # YuNet 검출기 생성
                    self.detector = cv2.FaceDetectorYN.create(
                        str(model_path),
                        "",  # config (불필요)
                        self.input_size,
                        score_threshold=self.min_detection_confidence,
                        nms_threshold=0.3,
                        top_k=5000
                    )
                    self.is_model_loaded = True
                    self.logger.info("YuNet 모델 로드 완료 (고정확도 모드)")
                else:
                    raise Exception("YuNet 모델 파일을 찾을 수 없습니다")

        except Exception as e:
            self.logger.error(f"YuNet 모델 로드 실패: {e}")
            self.is_model_loaded = False

    def _download_yunet_model(self) -> Path:
        """
        YuNet ONNX 모델 다운로드 (~2.8MB)
        """
        model_filename = "face_detection_yunet_2023mar.onnx"
        model_path = self.model_dir / model_filename

        if not model_path.exists():
            try:
                self.logger.info(f"YuNet 모델 다운로드 중: {model_filename}")
                url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
                urllib.request.urlretrieve(url, model_path)
                self.logger.info(f"다운로드 완료: {model_filename} (2.8MB)")
            except Exception as e:
                self.logger.error(f"YuNet 모델 다운로드 실패: {e}")

        return model_path
        
    def _unload_model(self):
        """
        YuNet 모델 언로드 (메모리 절약)
        """
        try:
            if self.is_model_loaded:
                self.detector = None
                self.is_model_loaded = False
                gc.collect()
                self.logger.info("YuNet 모델 언로드 완료 - 메모리 절약")
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

                    self.logger.info("얼굴 감지 활성화 (15초간 - YuNet 고정확도)")
    
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
        이미지에서 얼굴을 탐지 (YuNet 사용)
        메모리 효율성을 위해 탐지 시간에만 작동

        Args:
            image (np.ndarray): BGR 형식의 이미지
            force_detection (bool): 강제 탐지 모드 (테스트용)

        Returns:
            List[dict]: 탐지된 얼굴 정보 리스트
                       각 dict는 'box', 'confidence', 'keypoints' 키를 포함
                       keypoints: right_eye, left_eye, nose, right_mouth, left_mouth
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
                self.start_detection_cycle()

            # 모델이 로드되지 않았으면 빈 결과 반환
            if not self.is_model_loaded or self.detector is None:
                return []

            with self.detection_lock:
                if not self.detection_active and not force_detection:
                    return []

                # 이미지 크기 설정 (YuNet은 동적 입력 크기 지원)
                h, w = image.shape[:2]
                self.detector.setInputSize((w, h))

                # YuNet 추론 수행
                _, faces_raw = self.detector.detect(image)

                faces = []
                if faces_raw is not None:
                    for face_data in faces_raw:
                        # YuNet 출력 형식: [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm, score]
                        x, y, w_box, h_box = face_data[0:4].astype(int)
                        confidence = float(face_data[14])

                        # 랜드마크 5개 포인트 (YuNet의 강점!)
                        landmarks = {
                            'right_eye': (int(face_data[4]), int(face_data[5])),
                            'left_eye': (int(face_data[6]), int(face_data[7])),
                            'nose': (int(face_data[8]), int(face_data[9])),
                            'right_mouth': (int(face_data[10]), int(face_data[11])),
                            'left_mouth': (int(face_data[12]), int(face_data[13]))
                        }

                        # 박스 검증
                        if x >= 0 and y >= 0 and x + w_box <= w and y + h_box <= h:
                            faces.append({
                                'box': [x, y, w_box, h_box],
                                'confidence': confidence,
                                'keypoints': landmarks
                            })

                self.logger.debug(f"YuNet 탐지된 얼굴 수: {len(faces)}")
                return faces

        except Exception as e:
            self.logger.error(f"YuNet 얼굴 탐지 중 오류 발생: {e}")
            return []
    
    def has_faces(self, image: np.ndarray, confidence_threshold=0.7, force_detection: bool = False) -> bool:
        """
        이미지에 얼굴이 있는지 확인 (YuNet 사용)
        메모리 효율성을 위해 탐지 시간에만 실제 검사 수행

        Args:
            image (np.ndarray): BGR 형식의 이미지
            confidence_threshold (float): 얼굴 탐지 신뢰도 임계값 (YuNet 최적값: 0.7)
            force_detection (bool): 강제 탐지 모드

        Returns:
            bool: 얼굴이 탐지되면 True
        """
        if not force_detection and not self.should_activate_detection():
            return False

        faces = self.detect_faces(image, force_detection=force_detection)
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
            'last_detection': self.last_detection_time.isoformat() if self.last_detection_time else None,
            'model_type': 'YuNet (High Accuracy)'
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
            
            self.logger.info("YuNet 탐지기 리소스 정리 완료")
            
        except Exception as e:
            self.logger.error(f"리소스 정리 중 오류: {e}")
    
    def draw_faces(self, image: np.ndarray, save_path: Optional[str] = None, force: bool = False) -> np.ndarray:
        """
        탐지된 얼굴에 박스와 랜드마크를 그려서 시각화

        Args:
            image (np.ndarray): BGR 형식의 이미지
            save_path (str, optional): 저장할 경로
            force (bool): 강제 탐지 여부 (테스트용)

        Returns:
            np.ndarray: 얼굴 박스와 랜드마크가 그려진 이미지
        """
        if force:
            faces = self.force_detection(image)
        else:
            faces = self.detect_faces(image)

        result_image = image.copy()

        # 상태 정보 표시
        status = self.get_memory_status()
        status_text = f"YuNet: {'Loaded' if status['model_loaded'] else 'Unloaded'} | " \
                     f"Active: {'Yes' if status['detection_active'] else 'No'}"

        cv2.putText(result_image, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        for face in faces:
            x, y, w, h = face['box']
            confidence = face['confidence']

            # 얼굴 박스 (초록색)
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # 신뢰도 텍스트
            text = f"Face: {confidence:.2f}"
            cv2.putText(result_image, text, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 랜드마크 그리기 (YuNet의 장점!)
            if 'keypoints' in face and face['keypoints']:
                for point_name, (px, py) in face['keypoints'].items():
                    # 눈: 파란색, 코: 빨간색, 입: 주황색
                    if 'eye' in point_name:
                        color = (255, 0, 0)  # 파란색
                    elif 'nose' in point_name:
                        color = (0, 0, 255)  # 빨간색
                    else:  # mouth
                        color = (0, 165, 255)  # 주황색
                    cv2.circle(result_image, (px, py), 3, color, -1)

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
    logging.basicConfig(level=logging.INFO)

    detector = FaceDetector()

    print("YuNet 고정확도 얼굴 탐지기 테스트")
    print("=" * 50)
    print(f"초기 상태: {detector.get_memory_status()}")

    # 웹캠 테스트
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

                key = cv2.waitKey(1) & 0xFF

                if key == 27:  # ESC
                    break
                elif key == ord('f'):  # 강제 탐지
                    result = detector.draw_faces(frame, force=True)
                    print("강제 탐지 실행 (YuNet 고정확도)")
                else:
                    result = detector.draw_faces(frame)

                current_time = datetime.now().strftime("%H:%M:%S")
                cv2.putText(result, f"Time: {current_time}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                cv2.imshow("YuNet Face Detection (High Accuracy)", result)

            cap.release()
            cv2.destroyAllWindows()
        else:
            print("웹캠을 사용할 수 없습니다.")
    except Exception as e:
        print(f"웹캠 테스트 오류: {e}")

    detector.cleanup()
    print("\nYuNet 얼굴 탐지 모듈 테스트 완료")