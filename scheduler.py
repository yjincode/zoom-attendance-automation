"""
스케줄링 시스템
APScheduler를 사용하여 교시별 출석 체크 자동화
"""

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from datetime import datetime, time
import logging
import time as time_module
from typing import List, Tuple

class ClassScheduler:
    """
    교시별 스케줄링을 관리하는 클래스
    1교시 09:30 시작, 점심시간 12:30-14:30 제외, 총 8교시
    """
    
    def __init__(self, capture_callback=None):
        """
        스케줄러 초기화
        
        Args:
            capture_callback: 캡쳐 실행 콜백 함수
        """
        self.scheduler = BlockingScheduler()
        self.capture_callback = capture_callback
        self.logger = logging.getLogger(__name__)
        
        # 교시 시간표 정의 (시작시간, 종료시간)
        self.class_schedule = [
            (time(9, 30), time(10, 30)),   # 1교시
            (time(10, 30), time(11, 30)),  # 2교시  
            (time(11, 30), time(12, 30)),  # 3교시
            (time(12, 30), time(13, 30)),  # 4교시
            (time(14, 30), time(15, 30)),  # 5교시
            (time(15, 30), time(16, 30)),  # 6교시
            (time(16, 30), time(17, 30)),  # 7교시
            (time(17, 30), time(18, 30)),  # 8교시
        ]
        
        self.logger.info(f"총 {len(self.class_schedule)}교시 스케줄 설정 완료")
    
    def setup_capture_jobs(self):
        """
        각 교시별 캡쳐 작업을 스케줄러에 등록
        각 교시의 35분~40분 사이에 캡쳐 수행 (5분간)
        """
        for period, (start_time, end_time) in enumerate(self.class_schedule, 1):
            # 캡쳐 시작 시간 (교시 시작 + 35분)
            capture_start_hour = start_time.hour
            capture_start_minute = start_time.minute + 35
            
            # 60분 넘어가면 시간 조정
            if capture_start_minute >= 60:
                capture_start_hour += 1
                capture_start_minute -= 60
            
            # 캡쳐 종료 시간 (교시 시작 + 40분) - 5분간만
            capture_end_hour = start_time.hour
            capture_end_minute = start_time.minute + 40
            
            if capture_end_minute >= 60:
                capture_end_hour += 1
                capture_end_minute -= 60
            
            # 캡쳐 작업 스케줄 등록 (매분마다 실행)
            for minute in range(capture_start_minute, capture_end_minute + 1):
                hour = capture_start_hour
                if minute >= 60:
                    hour += 1
                    minute -= 60
                
                job_id = f"capture_period_{period}_time_{hour:02d}{minute:02d}"
                
                self.scheduler.add_job(
                    func=self._execute_capture,
                    trigger=CronTrigger(
                        hour=hour,
                        minute=minute,
                        second=0
                    ),
                    args=[period],
                    id=job_id,
                    max_instances=1,
                    replace_existing=True
                )
            
            self.logger.info(f"{period}교시 캡쳐 스케줄 등록: "
                           f"{capture_start_hour:02d}:{capture_start_minute:02d} ~ "
                           f"{capture_end_hour:02d}:{capture_end_minute:02d}")
    
    def _execute_capture(self, period: int):
        """
        캡쳐 실행 함수
        
        Args:
            period (int): 교시 번호
        """
        try:
            current_time = datetime.now().strftime("%H:%M:%S")
            self.logger.info(f"{period}교시 캡쳐 실행 - {current_time}")
            
            if self.capture_callback:
                self.capture_callback(period)
            else:
                self.logger.warning("캡쳐 콜백 함수가 설정되지 않음")
                
        except Exception as e:
            self.logger.error(f"{period}교시 캡쳐 실행 중 오류: {e}")
    
    def start(self):
        """
        스케줄러 시작
        """
        try:
            self.logger.info("스케줄러 시작")
            self.setup_capture_jobs()
            
            # 등록된 작업 목록 출력
            jobs = self.scheduler.get_jobs()
            self.logger.info(f"등록된 작업 수: {len(jobs)}")
            
            # 스케줄러 실행
            self.scheduler.start()
            
        except KeyboardInterrupt:
            self.logger.info("사용자에 의해 스케줄러 종료")
            self.stop()
        except Exception as e:
            self.logger.error(f"스케줄러 실행 중 오류: {e}")
            self.stop()
    
    def stop(self):
        """
        스케줄러 중지
        """
        try:
            self.scheduler.shutdown()
            self.logger.info("스케줄러가 정상적으로 종료됨")
        except Exception as e:
            self.logger.error(f"스케줄러 종료 중 오류: {e}")
    
    def get_next_capture_time(self) -> str:
        """
        다음 캡쳐 예정 시간 반환
        
        Returns:
            str: 다음 캡쳐 시간 문자열
        """
        try:
            jobs = self.scheduler.get_jobs()
            if jobs:
                next_job = min(jobs, key=lambda x: x.next_run_time)
                return next_job.next_run_time.strftime("%Y-%m-%d %H:%M:%S")
            else:
                return "예정된 작업 없음"
        except Exception as e:
            self.logger.error(f"다음 실행 시간 조회 오류: {e}")
            return "오류 발생"
    
    def is_class_time(self) -> Tuple[bool, int]:
        """
        현재 시간이 수업 시간인지 확인
        
        Returns:
            Tuple[bool, int]: (수업시간 여부, 교시 번호)
        """
        current_time = datetime.now().time()
        
        for period, (start_time, end_time) in enumerate(self.class_schedule, 1):
            if start_time <= current_time <= end_time:
                return True, period
        
        return False, 0
    
    def is_capture_time(self) -> Tuple[bool, int]:
        """
        현재 시간이 캡쳐 시간인지 확인 (교시의 35~40분, 5분간)
        
        Returns:
            Tuple[bool, int]: (캡쳐시간 여부, 교시 번호)
        """
        current_time = datetime.now().time()
        
        for period, (start_time, end_time) in enumerate(self.class_schedule, 1):
            # 캡쳐 시간 계산 (35-40분)
            capture_start_minute = start_time.minute + 35
            capture_end_minute = start_time.minute + 40
            
            capture_start_hour = start_time.hour
            capture_end_hour = start_time.hour
            
            if capture_start_minute >= 60:
                capture_start_hour += 1
                capture_start_minute -= 60
            
            if capture_end_minute >= 60:
                capture_end_hour += 1
                capture_end_minute -= 60
            
            capture_start = time(capture_start_hour, capture_start_minute)
            capture_end = time(capture_end_hour, capture_end_minute)
            
            if capture_start <= current_time <= capture_end:
                return True, period
        
        return False, 0

# 테스트용 캡쳐 콜백 함수
def test_capture_callback(period: int):
    """
    테스트용 캡쳐 콜백
    
    Args:
        period (int): 교시 번호
    """
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[테스트] {period}교시 캡쳐 실행됨 - {current_time}")

# 테스트 코드
if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 스케줄러 생성
    scheduler = ClassScheduler(capture_callback=test_capture_callback)
    
    # 현재 시간 정보
    now = datetime.now()
    print(f"현재 시간: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 수업 시간 확인
    is_class, class_period = scheduler.is_class_time()
    print(f"현재 수업 시간: {is_class}, 교시: {class_period}")
    
    # 캡쳐 시간 확인
    is_capture, capture_period = scheduler.is_capture_time()
    print(f"현재 캡쳐 시간: {is_capture}, 교시: {capture_period}")
    
    # 테스트 모드 (실제 스케줄러는 주석 해제)
    print("\n스케줄러 테스트 준비 완료")
    print("실제 실행하려면 scheduler.start() 호출")
    
    # 실제 스케줄러 시작 (테스트 시 주석 해제)
    # try:
    #     scheduler.start()
    # except KeyboardInterrupt:
    #     print("\n프로그램 종료")
    #     scheduler.stop()