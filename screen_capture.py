"""
화면 캡쳐 모듈
mss를 사용하여 전체 화면을 캡쳐하고 시스템 시계를 포함하는 기능을 제공
"""

import mss
import numpy as np
from PIL import Image
import cv2
from datetime import datetime
import os
from typing import Tuple, Optional
import logging
import threading

class ScreenCapture:
    """
    전체 화면 캡쳐를 담당하는 클래스
    시스템 시계가 포함된 브라우저 전체 화면을 캡쳐
    """
    
    def __init__(self, monitor_number: int = 2):
        """
        화면 캡쳐 초기화
        
        Args:
            monitor_number (int): 캡쳐할 모니터 번호 (1=메인, 2=서브)
        """
        self.monitor_number = monitor_number
        self.logger = logging.getLogger(__name__)
        
        # 스레드 로컬 저장소 (srcdc 오류 방지)
        self._local = threading.local()
        
        # 초기 모니터 정보 확인 (임시 mss 인스턴스 사용)
        with mss.mss() as temp_sct:
            monitors = temp_sct.monitors
            self.logger.info(f"사용 가능한 모니터 수: {len(monitors) - 1}")  # 0번은 전체 화면
            for i, monitor in enumerate(monitors):
                if i > 0:  # 0번 제외
                    self.logger.info(f"모니터 {i}: {monitor}")
            
            # 서브 모니터 확인
            if monitor_number > len(monitors) - 1:
                self.logger.warning(f"모니터 {monitor_number}가 없습니다. 모니터 1을 사용합니다.")
                self.monitor_number = 1
    
    def _get_sct(self):
        """
        스레드 로컬 mss 인스턴스 반환 (srcdc 오류 방지)
        각 스레드마다 독립적인 mss 인스턴스를 생성하여 Windows GDI 충돌 방지
        """
        if not hasattr(self._local, 'sct') or self._local.sct is None:
            self._local.sct = mss.mss()
            self.logger.debug(f"스레드 {threading.current_thread().name}에 새 mss 인스턴스 생성")
        return self._local.sct
    
    def capture_screen(self) -> np.ndarray:
        """
        전체 화면을 캡쳐하여 numpy 배열로 반환
        스레드 안전성을 위해 스레드 로컬 mss 인스턴스 사용
        
        Returns:
            np.ndarray: BGR 형식의 화면 이미지
        """
        try:
            # 스레드 로컬 mss 인스턴스 가져오기
            sct = self._get_sct()
            
            # 지정된 모니터의 화면 캡쳐
            monitor = sct.monitors[self.monitor_number]
            screenshot = sct.grab(monitor)
            
            # PIL Image로 변환
            img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
            
            # numpy 배열로 변환 (OpenCV 형식)
            img_array = np.array(img)
            
            # RGB to BGR 변환 (OpenCV 표준)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            self.logger.debug(f"화면 캡쳐 완료: {img_bgr.shape}")
            return img_bgr
            
        except Exception as e:
            self.logger.error(f"화면 캡쳐 중 오류 발생: {e}")
            # 스레드 로컬 mss 인스턴스 초기화 (재시도를 위해)
            if hasattr(self._local, 'sct'):
                try:
                    self._local.sct.close()
                except:
                    pass
                self._local.sct = None
            return np.array([])
    
    def save_screenshot(self, filepath: str) -> bool:
        """
        화면을 캡쳐하고 파일로 저장
        
        Args:
            filepath (str): 저장할 파일 경로
            
        Returns:
            bool: 저장 성공 여부
        """
        try:
            # 디렉토리가 없으면 생성
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # 화면 캡쳐
            screenshot = self.capture_screen()
            
            if screenshot.size == 0:
                return False
            
            # 파일 저장
            success = cv2.imwrite(filepath, screenshot)
            
            if success:
                self.logger.info(f"스크린샷 저장 완료: {filepath}")
            else:
                self.logger.error(f"스크린샷 저장 실패: {filepath}")
                
            return success
            
        except Exception as e:
            self.logger.error(f"스크린샷 저장 중 오류: {e}")
            return False
    
    def get_screen_info(self) -> dict:
        """
        현재 모니터의 정보를 반환
        
        Returns:
            dict: 모니터 정보 (width, height, top, left)
        """
        # 스레드 로컬 mss 인스턴스 사용
        sct = self._get_sct()
        monitor = sct.monitors[self.monitor_number]
        return {
            'width': monitor['width'],
            'height': monitor['height'],
            'top': monitor['top'],
            'left': monitor['left']
        }
    
    def cleanup(self):
        """
        리소스 정리 - 스레드 종료 시 호출
        """
        try:
            if hasattr(self._local, 'sct') and self._local.sct is not None:
                self._local.sct.close()
                self._local.sct = None
                self.logger.debug(f"스레드 {threading.current_thread().name}의 mss 인스턴스 정리 완료")
        except Exception as e:
            self.logger.error(f"mss 인스턴스 정리 중 오류: {e}")
    
    def change_monitor(self, monitor_number: int):
        """
        모니터 변경
        
        Args:
            monitor_number (int): 새로운 모니터 번호
        """
        old_monitor = self.monitor_number
        self.monitor_number = monitor_number
        
        # 변경 확인을 위해 임시 mss 인스턴스로 모니터 존재 확인
        try:
            with mss.mss() as temp_sct:
                if monitor_number > len(temp_sct.monitors) - 1:
                    self.logger.warning(f"모니터 {monitor_number}가 없습니다. 모니터 {old_monitor}를 계속 사용합니다.")
                    self.monitor_number = old_monitor
                else:
                    self.logger.info(f"모니터 변경: {old_monitor} → {monitor_number}")
                    # 기존 스레드 로컬 인스턴스 정리 (새 모니터 정보 로드를 위해)
                    self.cleanup()
        except Exception as e:
            self.logger.error(f"모니터 변경 중 오류: {e}")
            self.monitor_number = old_monitor

class ImageCandidate:
    """
    캡쳐된 이미지 후보를 관리하는 클래스
    선명도 기반으로 최적의 이미지를 선별
    """
    
    def __init__(self, max_candidates: int = 10):
        """
        이미지 후보 관리자 초기화
        
        Args:
            max_candidates (int): 저장할 최대 후보 수
        """
        self.max_candidates = max_candidates
        self.candidates = []  # [(image, sharpness, timestamp), ...]
        self.logger = logging.getLogger(__name__)
    
    def add_candidate(self, image: np.ndarray, timestamp: datetime) -> None:
        """
        이미지 후보를 추가
        
        Args:
            image (np.ndarray): BGR 형식의 이미지
            timestamp (datetime): 캡쳐 시각
        """
        from face_detector import calculate_image_sharpness
        
        # 선명도 계산
        sharpness = calculate_image_sharpness(image)
        
        # 후보 추가
        self.candidates.append((image.copy(), sharpness, timestamp))
        
        # 선명도 기준으로 정렬 (높은 순)
        self.candidates.sort(key=lambda x: x[1], reverse=True)
        
        # 최대 개수 유지
        if len(self.candidates) > self.max_candidates:
            self.candidates = self.candidates[:self.max_candidates]
        
        self.logger.debug(f"후보 이미지 추가: 선명도={sharpness:.2f}, 총 후보수={len(self.candidates)}")
    
    def save_best_candidates(self, output_dir: str, class_period: int, date_str: str) -> list:
        """
        가장 선명한 후보들을 파일로 저장
        
        Args:
            output_dir (str): 출력 디렉토리
            class_period (int): 교시 번호
            date_str (str): 날짜 문자열 (YYYYMMDD)
            
        Returns:
            list: 저장된 파일명 리스트
        """
        saved_files = []
        
        try:
            # 출력 디렉토리 생성
            os.makedirs(output_dir, exist_ok=True)
            
            # 선명도 순으로 저장
            for i, (image, sharpness, timestamp) in enumerate(self.candidates):
                filename = f"{date_str}_{class_period}교시_{i+1}.png"
                filepath = os.path.join(output_dir, filename)
                
                success = cv2.imwrite(filepath, image)
                if success:
                    saved_files.append(filename)
                    self.logger.info(f"이미지 저장: {filename} (선명도: {sharpness:.2f})")
                else:
                    self.logger.error(f"이미지 저장 실패: {filename}")
            
            self.logger.info(f"{class_period}교시 총 {len(saved_files)}개 이미지 저장 완료")
            
        except Exception as e:
            self.logger.error(f"후보 이미지 저장 중 오류: {e}")
        
        return saved_files
    
    def clear_candidates(self) -> None:
        """
        모든 후보 이미지 삭제
        """
        self.candidates.clear()
        self.logger.debug("후보 이미지 목록 초기화")
    
    def get_candidate_count(self) -> int:
        """
        현재 후보 이미지 수 반환
        
        Returns:
            int: 후보 이미지 수
        """
        return len(self.candidates)

# 테스트 코드
if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    
    # 화면 캡쳐 테스트
    capturer = ScreenCapture()
    
    # 화면 정보 출력
    screen_info = capturer.get_screen_info()
    print(f"화면 정보: {screen_info}")
    
    # 테스트 캡쳐
    image = capturer.capture_screen()
    if image.size > 0:
        print(f"캡쳐 성공: {image.shape}")
        
        # 테스트 저장
        test_dir = "test_captures"
        os.makedirs(test_dir, exist_ok=True)
        test_file = os.path.join(test_dir, "test_capture.png")
        
        if capturer.save_screenshot(test_file):
            print(f"테스트 이미지 저장 완료: {test_file}")
        else:
            print("테스트 이미지 저장 실패")
    else:
        print("캡쳐 실패")
    
    # 후보 관리자 테스트
    candidate_manager = ImageCandidate(max_candidates=3)
    
    if image.size > 0:
        # 몇 개의 테스트 후보 추가
        for i in range(5):
            candidate_manager.add_candidate(image, datetime.now())
        
        print(f"후보 수: {candidate_manager.get_candidate_count()}")
        
        # 테스트 저장
        saved_files = candidate_manager.save_best_candidates(
            test_dir, 1, datetime.now().strftime("%Y%m%d")
        )
        print(f"저장된 파일: {saved_files}")
    
    print("화면 캡쳐 모듈 테스트 완료")