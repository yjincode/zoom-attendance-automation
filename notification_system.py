"""
알림 시스템 모듈
교시별 시작/종료 알림과 출석 체크 상태 알림을 제공
"""

import os
import platform
from plyer import notification
import logging
from datetime import datetime
from typing import Optional

class NotificationSystem:
    """
    크로스 플랫폼 알림 시스템
    교시 시작, 캡쳐 시작/종료, 결과 등을 알림
    """
    
    def __init__(self, app_name: str = "Zoom 출석 체크"):
        """
        알림 시스템 초기화
        
        Args:
            app_name (str): 애플리케이션 이름
        """
        self.app_name = app_name
        self.logger = logging.getLogger(__name__)
        self.system = platform.system()
        
        # 아이콘 경로 설정
        self.icon_path = self._get_icon_path()
        
        self.logger.info(f"알림 시스템 초기화 완료 (OS: {self.system})")
    
    def _get_icon_path(self) -> Optional[str]:
        """
        알림 아이콘 경로 반환
        
        Returns:
            str: 아이콘 파일 경로
        """
        # 프로젝트 디렉토리에서 아이콘 찾기
        possible_icons = [
            "icon.png",
            "icon.ico",
            "assets/icon.png",
            "resources/icon.png"
        ]
        
        for icon in possible_icons:
            if os.path.exists(icon):
                return os.path.abspath(icon)
        
        return None
    
    def _send_notification(self, title: str, message: str, timeout: int = 5):
        """
        실제 알림 전송
        
        Args:
            title (str): 알림 제목
            message (str): 알림 내용
            timeout (int): 알림 표시 시간 (초)
        """
        try:
            notification.notify(
                title=title,
                message=message,
                app_name=self.app_name,
                app_icon=self.icon_path,
                timeout=timeout
            )
            self.logger.info(f"알림 전송: {title} - {message}")
            
        except Exception as e:
            self.logger.error(f"알림 전송 실패: {e}")
            # 콘솔에라도 출력
            print(f"🔔 {title}: {message}")
    
    def notify_class_start(self, period: int, start_time: str):
        """
        교시 시작 알림
        
        Args:
            period (int): 교시 번호
            start_time (str): 시작 시간
        """
        title = f"{period}교시 시작"
        message = f"{start_time}에 {period}교시가 시작되었습니다.\n35분 후 출석 체크를 시작합니다."
        
        self._send_notification(title, message, timeout=7)
    
    def notify_capture_start(self, period: int):
        """
        캡쳐 시작 알림
        
        Args:
            period (int): 교시 번호
        """
        title = f"{period}교시 출석 체크 시작"
        message = f"{period}교시 출석 확인을 시작합니다.\n10분간 자동으로 화면을 캡쳐합니다."
        
        self._send_notification(title, message, timeout=6)
    
    def notify_capture_end(self, period: int, capture_count: int):
        """
        캡쳐 종료 알림
        
        Args:
            period (int): 교시 번호
            capture_count (int): 캡쳐된 이미지 수
        """
        title = f"{period}교시 출석 체크 완료"
        
        if capture_count > 0:
            message = f"{period}교시 출석 확인 완료!\n{capture_count}개의 이미지가 저장되었습니다."
        else:
            message = f"{period}교시 출석 확인 실패.\n얼굴이 감지된 이미지가 없습니다."
        
        self._send_notification(title, message, timeout=8)
    
    def notify_face_detected(self, period: int, participant_count: int):
        """
        실시간 얼굴 감지 알림
        
        Args:
            period (int): 교시 번호
            participant_count (int): 감지된 참가자 수
        """
        title = f"참가자 감지"
        message = f"{period}교시: {participant_count}명 참가 확인"
        
        # 짧은 시간 표시 (너무 자주 뜨지 않도록)
        self._send_notification(title, message, timeout=3)
    
    def notify_no_faces(self, period: int):
        """
        얼굴 미감지 경고 알림
        
        Args:
            period (int): 교시 번호
        """
        title = f"⚠️ 참가자 미감지"
        message = f"{period}교시: 화면에서 참가자를 찾을 수 없습니다.\nZoom 화면을 확인해주세요."
        
        self._send_notification(title, message, timeout=6)
    
    def notify_system_start(self):
        """
        시스템 시작 알림
        """
        title = "출석 자동화 시스템 시작"
        message = "Zoom 출석 자동화가 시작되었습니다.\n각 교시 35분부터 자동으로 출석을 체크합니다."
        
        self._send_notification(title, message, timeout=8)
    
    def notify_system_stop(self):
        """
        시스템 종료 알림
        """
        title = "출석 자동화 시스템 종료"
        message = "Zoom 출석 자동화가 종료되었습니다.\n오늘의 출석 기록이 저장되었습니다."
        
        self._send_notification(title, message, timeout=6)
    
    def notify_error(self, error_message: str):
        """
        오류 발생 알림
        
        Args:
            error_message (str): 오류 메시지
        """
        title = "❌ 시스템 오류"
        message = f"오류가 발생했습니다:\n{error_message[:100]}..."
        
        self._send_notification(title, message, timeout=10)
    
    def notify_monitor_switched(self, monitor_number: int):
        """
        모니터 전환 알림
        
        Args:
            monitor_number (int): 새로운 모니터 번호
        """
        title = "모니터 전환"
        message = f"모니터 {monitor_number}로 전환했습니다.\nZoom이 이 모니터에 있는지 확인해주세요."
        
        self._send_notification(title, message, timeout=6)
    
    def notify_zoom_not_found(self):
        """
        Zoom 미발견 알림
        """
        title = "⚠️ Zoom 미발견"
        message = "화면에서 Zoom을 찾을 수 없습니다.\nZoom이 실행 중이고 화면에 표시되는지 확인해주세요."
        
        self._send_notification(title, message, timeout=8)

class SoundNotification:
    """
    소리 알림 시스템 (간단한 비프음)
    """
    
    def __init__(self):
        """
        소리 알림 초기화
        """
        self.system = platform.system()
        self.logger = logging.getLogger(__name__)
    
    def beep(self, count: int = 1, interval: float = 0.5):
        """
        비프음 재생
        
        Args:
            count (int): 비프음 횟수
            interval (float): 비프음 간격 (초)
        """
        try:
            import time
            
            for _ in range(count):
                if self.system == "Windows":
                    import winsound
                    winsound.Beep(1000, 300)  # 1000Hz, 300ms
                elif self.system == "Darwin":  # macOS
                    os.system("afplay /System/Library/Sounds/Ping.aiff")
                else:  # Linux
                    os.system("beep -f 1000 -l 300")
                
                if count > 1:
                    time.sleep(interval)
                    
        except Exception as e:
            self.logger.error(f"소리 알림 실패: {e}")
    
    def success_sound(self):
        """성공 알림음 (2번 비프)"""
        self.beep(2, 0.3)
    
    def warning_sound(self):
        """경고 알림음 (3번 비프)"""
        self.beep(3, 0.2)
    
    def error_sound(self):
        """오류 알림음 (길게 1번)"""
        self.beep(1)

# 테스트 코드
if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    
    # 알림 시스템 테스트
    notifier = NotificationSystem()
    sound = SoundNotification()
    
    print("알림 시스템 테스트 시작...")
    
    # 각종 알림 테스트
    notifier.notify_system_start()
    sound.success_sound()
    
    import time
    time.sleep(2)
    
    notifier.notify_class_start(1, "09:30")
    time.sleep(2)
    
    notifier.notify_capture_start(1)
    time.sleep(2)
    
    notifier.notify_face_detected(1, 5)
    time.sleep(2)
    
    notifier.notify_capture_end(1, 8)
    sound.success_sound()
    time.sleep(2)
    
    notifier.notify_no_faces(2)
    sound.warning_sound()
    time.sleep(2)
    
    notifier.notify_system_stop()
    
    print("알림 시스템 테스트 완료")