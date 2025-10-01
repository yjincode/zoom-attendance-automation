"""
Zoom 강의 출석 자동화 메인 프로그램
모든 모듈을 통합하여 자동 출석 체크 시스템을 실행
"""

import os
import sys
from datetime import datetime
import logging
import signal
import time

# 자체 모듈 import
from face_detector import FaceDetector
from screen_capture import ScreenCapture, ImageCandidate
from scheduler import ClassScheduler
from logger import AttendanceLogger

class ZoomAttendanceSystem:
    """
    Zoom 강의 출석 자동화 시스템 메인 클래스
    모든 모듈을 통합하여 관리
    """
    
    def __init__(self, output_dir: str = "captures", log_file: str = "attendance_log.csv"):
        """
        시스템 초기화
        
        Args:
            output_dir (str): 캡쳐 이미지 저장 디렉토리
            log_file (str): 로그 파일 경로
        """
        self.output_dir = output_dir
        self.log_file = log_file
        
        # 로깅 설정
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # 각 모듈 초기화
        self.face_detector = FaceDetector(min_face_size=30)
        self.screen_capturer = ScreenCapture(monitor_number=1)
        self.attendance_logger = AttendanceLogger(log_file)
        
        # 교시별 이미지 후보 관리자
        self.current_candidates = ImageCandidate(max_candidates=10)
        self.current_period = 0
        
        # 스케줄러는 나중에 초기화
        self.scheduler = None
        
        self.logger.info("Zoom 출석 자동화 시스템 초기화 완료")
        
        # 종료 신호 처리
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _setup_logging(self):
        """
        로깅 설정
        """
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter(log_format))
        
        # 파일 핸들러
        file_handler = logging.FileHandler('zoom_attendance.log', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(log_format))
        
        # 루트 로거 설정
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)
    
    def _signal_handler(self, signum, frame):
        """
        종료 신호 처리
        """
        self.logger.info(f"종료 신호 수신: {signum}")
        self.shutdown()
        sys.exit(0)
    
    def capture_and_process(self, period: int):
        """
        화면 캡쳐 및 얼굴 감지 처리
        
        Args:
            period (int): 교시 번호
        """
        try:
            self.logger.info(f"{period}교시 캡쳐 및 처리 시작")
            
            # 교시가 바뀌면 이전 교시 결과 저장
            if self.current_period != period and self.current_period > 0:
                self._save_period_results(self.current_period)
            
            self.current_period = period
            
            # 화면 캡쳐
            screenshot = self.screen_capturer.capture_screen()
            
            if screenshot.size == 0:
                self.logger.error(f"{period}교시 화면 캡쳐 실패")
                return
            
            # 얼굴 감지
            has_faces = self.face_detector.has_faces(screenshot, confidence_threshold=0.8)
            
            if has_faces:
                # 얼굴이 감지되면 후보에 추가
                self.current_candidates.add_candidate(screenshot, datetime.now())
                self.logger.info(f"{period}교시 얼굴 감지 성공 - 후보 추가 "
                               f"(현재 후보 수: {self.current_candidates.get_candidate_count()})")
            else:
                self.logger.debug(f"{period}교시 얼굴 미감지")
                
        except Exception as e:
            self.logger.error(f"{period}교시 캡쳐 처리 중 오류: {e}")
    
    def _save_period_results(self, period: int):
        """
        교시별 결과 저장
        
        Args:
            period (int): 교시 번호
        """
        try:
            current_date = datetime.now().strftime("%Y%m%d")
            date_str = datetime.now().strftime("%Y-%m-%d")
            
            # 후보 이미지들을 저장
            saved_files = self.current_candidates.save_best_candidates(
                output_dir=self.output_dir,
                class_period=period,
                date_str=current_date
            )
            
            # 로그 기록
            status = "success" if saved_files else "failed"
            self.attendance_logger.log_attendance(
                date=date_str,
                class_period=period,
                capture_count=len(saved_files),
                file_names=saved_files,
                status=status
            )
            
            self.logger.info(f"{period}교시 결과 저장 완료: {len(saved_files)}개 이미지")
            
            # 후보 목록 초기화
            self.current_candidates.clear_candidates()
            
        except Exception as e:
            self.logger.error(f"{period}교시 결과 저장 중 오류: {e}")
    
    def start(self):
        """
        시스템 시작
        """
        try:
            self.logger.info("=== Zoom 출석 자동화 시스템 시작 ===")
            
            # 시스템 정보 출력
            screen_info = self.screen_capturer.get_screen_info()
            self.logger.info(f"모니터 정보: {screen_info}")
            
            # 스케줄러 생성 및 시작
            self.scheduler = ClassScheduler(capture_callback=self.capture_and_process)
            
            self.logger.info("스케줄러 시작 - 각 교시 35~45분에 자동 캡쳐 수행")
            self.scheduler.start()
            
        except KeyboardInterrupt:
            self.logger.info("사용자에 의해 시스템 종료")
            self.shutdown()
        except Exception as e:
            self.logger.error(f"시스템 실행 중 오류: {e}")
            self.shutdown()
    
    def shutdown(self):
        """
        시스템 종료
        """
        try:
            self.logger.info("시스템 종료 중...")
            
            # 현재 교시 결과가 있으면 저장
            if self.current_period > 0 and self.current_candidates.get_candidate_count() > 0:
                self._save_period_results(self.current_period)
            
            # 스케줄러 종료
            if self.scheduler:
                self.scheduler.stop()
            
            self.logger.info("시스템 종료 완료")
            
        except Exception as e:
            self.logger.error(f"시스템 종료 중 오류: {e}")
    
    def test_mode(self):
        """
        테스트 모드 - 즉시 캡쳐 및 처리 테스트
        """
        self.logger.info("=== 테스트 모드 시작 ===")
        
        try:
            # 테스트 캡쳐
            screenshot = self.screen_capturer.capture_screen()
            
            if screenshot.size == 0:
                self.logger.error("테스트 캡쳐 실패")
                return False
            
            # 테스트 이미지 저장
            test_file = os.path.join(self.output_dir, "test_capture.png")
            self.screen_capturer.save_screenshot(test_file)
            
            # 얼굴 감지 테스트
            faces = self.face_detector.detect_faces(screenshot)
            has_faces = len(faces) > 0
            
            # 결과 출력
            self.logger.info(f"테스트 결과:")
            self.logger.info(f"- 화면 캡쳐: 성공 ({screenshot.shape})")
            self.logger.info(f"- 얼굴 감지: {'성공' if has_faces else '실패'} ({len(faces)}개 감지)")
            self.logger.info(f"- 테스트 이미지: {test_file}")
            
            if has_faces:
                # 얼굴이 있는 경우 시각화 이미지 저장
                face_image = self.face_detector.draw_faces(
                    screenshot, 
                    os.path.join(self.output_dir, "test_face_detection.png")
                )
                self.logger.info("- 얼굴 감지 결과 이미지 저장 완료")
            
            return True
            
        except Exception as e:
            self.logger.error(f"테스트 모드 오류: {e}")
            return False

def main():
    """
    메인 함수
    """
    # 시스템 생성
    system = ZoomAttendanceSystem(
        output_dir="captures",
        log_file="attendance_log.csv"
    )
    
    # 명령행 인수 확인
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # 테스트 모드
        success = system.test_mode()
        if success:
            print("테스트 완료 - captures 폴더를 확인하세요")
        else:
            print("테스트 실패 - 로그를 확인하세요")
    else:
        # 일반 모드
        print("Zoom 출석 자동화 시스템을 시작합니다...")
        print("각 교시 35~45분에 자동으로 캡쳐를 수행합니다.")
        print("종료하려면 Ctrl+C를 누르세요.")
        system.start()

if __name__ == "__main__":
    main()