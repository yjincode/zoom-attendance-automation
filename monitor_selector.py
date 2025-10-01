"""
모니터 선택 및 관리 모듈
듀얼 모니터 환경에서 적절한 모니터를 선택하고 관리
"""

import mss
import cv2
import numpy as np
from typing import List, Dict, Tuple
import logging

class MonitorManager:
    """
    멀티 모니터 환경을 관리하는 클래스
    """
    
    def __init__(self):
        """
        모니터 관리자 초기화
        """
        self.sct = mss.mss()
        self.monitors = self.sct.monitors
        self.logger = logging.getLogger(__name__)
        self._log_monitor_info()
    
    def _log_monitor_info(self):
        """
        사용 가능한 모니터 정보를 로깅
        """
        self.logger.info(f"총 모니터 수: {len(self.monitors) - 1}")  # 0번 제외
        
        for i, monitor in enumerate(self.monitors):
            if i == 0:
                self.logger.info(f"모니터 0 (전체): {monitor}")
            else:
                self.logger.info(f"모니터 {i}: {monitor}")
    
    def get_monitor_count(self) -> int:
        """
        사용 가능한 모니터 수 반환 (0번 제외)
        
        Returns:
            int: 모니터 수
        """
        return len(self.monitors) - 1
    
    def get_monitor_info(self, monitor_number: int) -> Dict:
        """
        특정 모니터의 정보 반환
        
        Args:
            monitor_number (int): 모니터 번호 (1부터 시작)
            
        Returns:
            dict: 모니터 정보
        """
        if monitor_number < 1 or monitor_number >= len(self.monitors):
            raise ValueError(f"잘못된 모니터 번호: {monitor_number}")
        
        monitor = self.monitors[monitor_number]
        return {
            'number': monitor_number,
            'width': monitor['width'],
            'height': monitor['height'],
            'left': monitor['left'],
            'top': monitor['top']
        }
    
    def list_all_monitors(self) -> List[Dict]:
        """
        모든 모니터 정보 리스트 반환
        
        Returns:
            List[Dict]: 모니터 정보 리스트
        """
        monitor_list = []
        for i in range(1, len(self.monitors)):
            try:
                monitor_info = self.get_monitor_info(i)
                monitor_list.append(monitor_info)
            except ValueError:
                continue
        
        return monitor_list
    
    def capture_monitor_preview(self, monitor_number: int, scale_factor: float = 0.3) -> np.ndarray:
        """
        모니터 미리보기 캡쳐 (작은 크기)
        
        Args:
            monitor_number (int): 모니터 번호
            scale_factor (float): 크기 조절 비율
            
        Returns:
            np.ndarray: 미리보기 이미지
        """
        try:
            monitor = self.monitors[monitor_number]
            screenshot = self.sct.grab(monitor)
            
            # PIL Image로 변환 후 numpy 배열로
            from PIL import Image
            img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
            img_array = np.array(img)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # 크기 조절
            height, width = img_bgr.shape[:2]
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            
            resized = cv2.resize(img_bgr, (new_width, new_height))
            return resized
            
        except Exception as e:
            self.logger.error(f"모니터 {monitor_number} 미리보기 캡쳐 실패: {e}")
            return np.array([])
    
    def find_zoom_monitor(self) -> int:
        """
        Zoom이 실행 중인 모니터를 찾기 (휴리스틱 방법)
        
        Returns:
            int: Zoom이 있을 가능성이 높은 모니터 번호
        """
        # 각 모니터를 캡쳐해서 Zoom 특징을 찾기
        for monitor_num in range(1, len(self.monitors)):
            try:
                preview = self.capture_monitor_preview(monitor_num, 0.2)
                if preview.size == 0:
                    continue
                
                # Zoom 특징 감지 (간단한 휴리스틱)
                # 1. 어두운 배경 (Zoom의 기본 배경)
                # 2. 네모난 영역들 (참가자 박스들)
                
                gray = cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY)
                
                # 어두운 픽셀 비율 계산
                dark_pixels = np.sum(gray < 50)
                total_pixels = gray.size
                dark_ratio = dark_pixels / total_pixels
                
                # 사각형 영역 감지
                edges = cv2.Canny(gray, 50, 150)
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # 적당한 크기의 사각형 개수
                rect_count = 0
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if 500 < area < 50000:  # 적당한 크기
                        # 사각형인지 확인
                        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
                        if len(approx) >= 4:
                            rect_count += 1
                
                self.logger.debug(f"모니터 {monitor_num}: 어두운 비율={dark_ratio:.2f}, 사각형 수={rect_count}")
                
                # Zoom 가능성 점수 계산
                zoom_score = dark_ratio * 0.7 + (rect_count / 10) * 0.3
                
                if zoom_score > 0.3 and rect_count >= 2:
                    self.logger.info(f"모니터 {monitor_num}에서 Zoom 감지됨 (점수: {zoom_score:.2f})")
                    return monitor_num
                    
            except Exception as e:
                self.logger.error(f"모니터 {monitor_num} Zoom 감지 실패: {e}")
                continue
        
        # 감지 실패 시 서브 모니터(2번) 기본 반환
        if len(self.monitors) > 2:
            self.logger.info("Zoom 자동 감지 실패, 모니터 2 사용")
            return 2
        else:
            self.logger.info("서브 모니터 없음, 모니터 1 사용")
            return 1
    
    def get_primary_monitor(self) -> int:
        """
        주 모니터 번호 반환
        
        Returns:
            int: 주 모니터 번호
        """
        # 대부분의 경우 모니터 1이 주 모니터
        return 1
    
    def get_secondary_monitor(self) -> int:
        """
        보조 모니터 번호 반환
        
        Returns:
            int: 보조 모니터 번호 (없으면 주 모니터)
        """
        if len(self.monitors) > 2:
            return 2
        else:
            return 1

# 사용 예시 및 테스트
if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    
    # 모니터 관리자 생성
    monitor_manager = MonitorManager()
    
    # 모든 모니터 정보 출력
    monitors = monitor_manager.list_all_monitors()
    print("사용 가능한 모니터:")
    for monitor in monitors:
        print(f"  모니터 {monitor['number']}: {monitor['width']}x{monitor['height']} "
              f"at ({monitor['left']}, {monitor['top']})")
    
    # Zoom 모니터 자동 감지
    zoom_monitor = monitor_manager.find_zoom_monitor()
    print(f"\nZoom이 있을 것으로 예상되는 모니터: {zoom_monitor}")
    
    # 각 모니터 미리보기 저장 (테스트용)
    for i in range(1, monitor_manager.get_monitor_count() + 1):
        preview = monitor_manager.capture_monitor_preview(i)
        if preview.size > 0:
            cv2.imwrite(f"monitor_{i}_preview.png", preview)
            print(f"모니터 {i} 미리보기 저장: monitor_{i}_preview.png")
    
    print("모니터 관리자 테스트 완료")