"""
로그 기록 시스템
CSV 파일로 출석 체크 결과를 기록하고 관리
"""

import pandas as pd
import os
from datetime import datetime
from typing import List, Optional
import logging

class AttendanceLogger:
    """
    출석 체크 결과를 CSV 파일로 기록하는 클래스
    """
    
    def __init__(self, log_file: str = "attendance_log.csv"):
        """
        로거 초기화
        
        Args:
            log_file (str): 로그 파일 경로
        """
        self.log_file = log_file
        self.logger = logging.getLogger(__name__)
        
        # CSV 파일이 없으면 헤더와 함께 생성
        self._initialize_log_file()
    
    def _initialize_log_file(self):
        """
        로그 파일 초기화 (헤더 생성)
        """
        if not os.path.exists(self.log_file):
            # 빈 DataFrame 생성
            df = pd.DataFrame(columns=[
                'date',           # 날짜 (YYYY-MM-DD)
                'class_period',   # 교시
                'capture_count',  # 저장된 이미지 개수
                'file_names',     # 저장된 파일명들 (세미콜론으로 구분)
                'timestamp',      # 기록 시각
                'status'          # 상태 (success/failed)
            ])
            
            # CSV 파일로 저장
            df.to_csv(self.log_file, index=False, encoding='utf-8-sig')
            self.logger.info(f"로그 파일 생성: {self.log_file}")
    
    def log_attendance(self, 
                      date: str, 
                      class_period: int, 
                      capture_count: int, 
                      file_names: List[str], 
                      status: str = "success") -> bool:
        """
        출석 체크 결과를 로그에 기록
        
        Args:
            date (str): 날짜 (YYYY-MM-DD)
            class_period (int): 교시
            capture_count (int): 저장된 이미지 개수
            file_names (List[str]): 저장된 파일명 리스트
            status (str): 상태 ("success" 또는 "failed")
            
        Returns:
            bool: 기록 성공 여부
        """
        try:
            # 새로운 레코드 생성
            new_record = {
                'date': date,
                'class_period': class_period,
                'capture_count': capture_count,
                'file_names': ';'.join(file_names) if file_names else '',
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'status': status
            }
            
            # 기존 데이터 읽기
            try:
                df = pd.read_csv(self.log_file, encoding='utf-8-sig')
            except (FileNotFoundError, pd.errors.EmptyDataError):
                # 파일이 없거나 비어있으면 새로 생성
                df = pd.DataFrame(columns=new_record.keys())
            
            # 새 레코드 추가
            df = pd.concat([df, pd.DataFrame([new_record])], ignore_index=True)
            
            # CSV 파일로 저장
            df.to_csv(self.log_file, index=False, encoding='utf-8-sig')
            
            self.logger.info(f"출석 로그 기록: {date} {class_period}교시 - {capture_count}개 이미지")
            return True
            
        except Exception as e:
            self.logger.error(f"로그 기록 중 오류: {e}")
            return False
    
    def get_daily_summary(self, date: str) -> dict:
        """
        특정 날짜의 출석 체크 요약 정보 반환
        
        Args:
            date (str): 조회할 날짜 (YYYY-MM-DD)
            
        Returns:
            dict: 요약 정보
        """
        try:
            df = pd.read_csv(self.log_file, encoding='utf-8-sig')
            daily_data = df[df['date'] == date]
            
            if daily_data.empty:
                return {
                    'date': date,
                    'total_periods': 0,
                    'total_captures': 0,
                    'successful_periods': 0,
                    'failed_periods': 0,
                    'periods': []
                }
            
            # 요약 정보 계산
            summary = {
                'date': date,
                'total_periods': len(daily_data),
                'total_captures': daily_data['capture_count'].sum(),
                'successful_periods': len(daily_data[daily_data['status'] == 'success']),
                'failed_periods': len(daily_data[daily_data['status'] == 'failed']),
                'periods': []
            }
            
            # 교시별 정보
            for _, row in daily_data.iterrows():
                period_info = {
                    'class_period': row['class_period'],
                    'capture_count': row['capture_count'],
                    'status': row['status'],
                    'timestamp': row['timestamp'],
                    'file_names': row['file_names'].split(';') if row['file_names'] else []
                }
                summary['periods'].append(period_info)
            
            return summary
            
        except Exception as e:
            self.logger.error(f"일일 요약 조회 중 오류: {e}")
            return {}
    
    def get_weekly_summary(self, start_date: str, end_date: str) -> dict:
        """
        특정 기간의 출석 체크 요약 정보 반환
        
        Args:
            start_date (str): 시작 날짜 (YYYY-MM-DD)
            end_date (str): 종료 날짜 (YYYY-MM-DD)
            
        Returns:
            dict: 주간 요약 정보
        """
        try:
            df = pd.read_csv(self.log_file, encoding='utf-8-sig')
            
            # 날짜 필터링
            mask = (df['date'] >= start_date) & (df['date'] <= end_date)
            period_data = df[mask]
            
            if period_data.empty:
                return {
                    'start_date': start_date,
                    'end_date': end_date,
                    'total_days': 0,
                    'total_periods': 0,
                    'total_captures': 0,
                    'daily_summaries': []
                }
            
            # 주간 요약
            summary = {
                'start_date': start_date,
                'end_date': end_date,
                'total_days': period_data['date'].nunique(),
                'total_periods': len(period_data),
                'total_captures': period_data['capture_count'].sum(),
                'successful_periods': len(period_data[period_data['status'] == 'success']),
                'failed_periods': len(period_data[period_data['status'] == 'failed']),
                'daily_summaries': []
            }
            
            # 일별 요약
            for date in period_data['date'].unique():
                daily_summary = self.get_daily_summary(date)
                summary['daily_summaries'].append(daily_summary)
            
            return summary
            
        except Exception as e:
            self.logger.error(f"주간 요약 조회 중 오류: {e}")
            return {}
    
    def export_to_excel(self, output_file: str) -> bool:
        """
        로그 데이터를 Excel 파일로 내보내기
        
        Args:
            output_file (str): 출력 파일 경로 (.xlsx)
            
        Returns:
            bool: 내보내기 성공 여부
        """
        try:
            df = pd.read_csv(self.log_file, encoding='utf-8-sig')
            
            # Excel 파일로 저장
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                # 전체 데이터
                df.to_excel(writer, sheet_name='전체_로그', index=False)
                
                # 날짜별 요약
                if not df.empty:
                    daily_summary = df.groupby('date').agg({
                        'class_period': 'count',
                        'capture_count': 'sum',
                        'status': lambda x: (x == 'success').sum()
                    }).rename(columns={
                        'class_period': '총_교시수',
                        'capture_count': '총_캡쳐수',
                        'status': '성공_교시수'
                    })
                    daily_summary.to_excel(writer, sheet_name='날짜별_요약')
            
            self.logger.info(f"Excel 파일 생성 완료: {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Excel 내보내기 중 오류: {e}")
            return False
    
    def clean_old_logs(self, days_to_keep: int = 30) -> bool:
        """
        오래된 로그 데이터 정리
        
        Args:
            days_to_keep (int): 보관할 일수
            
        Returns:
            bool: 정리 성공 여부
        """
        try:
            df = pd.read_csv(self.log_file, encoding='utf-8-sig')
            
            # 현재 날짜에서 지정된 일수만큼 빼기
            cutoff_date = (datetime.now() - pd.Timedelta(days=days_to_keep)).strftime('%Y-%m-%d')
            
            # 최근 데이터만 필터링
            df_filtered = df[df['date'] >= cutoff_date]
            
            # 삭제된 레코드 수 확인
            deleted_count = len(df) - len(df_filtered)
            
            if deleted_count > 0:
                # 필터링된 데이터로 파일 덮어쓰기
                df_filtered.to_csv(self.log_file, index=False, encoding='utf-8-sig')
                self.logger.info(f"오래된 로그 {deleted_count}개 레코드 삭제 완료")
            else:
                self.logger.info("삭제할 오래된 로그가 없음")
            
            return True
            
        except Exception as e:
            self.logger.error(f"로그 정리 중 오류: {e}")
            return False

# 테스트 코드
if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    
    # 테스트용 로거 생성
    test_logger = AttendanceLogger("test_attendance_log.csv")
    
    # 테스트 데이터 추가
    today = datetime.now().strftime('%Y-%m-%d')
    
    # 성공 케이스
    test_logger.log_attendance(
        date=today,
        class_period=1,
        capture_count=8,
        file_names=["20241001_1교시_1.png", "20241001_1교시_2.png"],
        status="success"
    )
    
    # 실패 케이스
    test_logger.log_attendance(
        date=today,
        class_period=2,
        capture_count=0,
        file_names=[],
        status="failed"
    )
    
    # 일일 요약 조회
    daily_summary = test_logger.get_daily_summary(today)
    print("일일 요약:")
    print(f"- 날짜: {daily_summary['date']}")
    print(f"- 총 교시: {daily_summary['total_periods']}")
    print(f"- 총 캡쳐: {daily_summary['total_captures']}")
    print(f"- 성공: {daily_summary['successful_periods']}")
    print(f"- 실패: {daily_summary['failed_periods']}")
    
    # Excel 내보내기 테스트
    excel_file = "test_attendance_report.xlsx"
    if test_logger.export_to_excel(excel_file):
        print(f"Excel 파일 생성됨: {excel_file}")
    
    print("로그 시스템 테스트 완료")