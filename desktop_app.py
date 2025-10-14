"""
Zoom 출석 자동화 데스크톱 애플리케이션
PyQt5를 사용한 GUI 버전
"""

import sys
import os
from datetime import datetime
import logging
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QLabel, QPushButton, QComboBox, 
                           QTextEdit, QGroupBox, QGridLayout, QFrame,
                           QSystemTrayIcon, QMenu, QAction, QMessageBox,
                           QCheckBox, QSpinBox, QSlider, QTabWidget)
from PyQt5.QtCore import QTimer, QThread, pyqtSignal, Qt, QSettings
from PyQt5.QtGui import QPixmap, QImage, QIcon, QFont
import json

# 자체 모듈 import
from screen_capture import ScreenCapture
from monitor_selector import MonitorManager
from zoom_detector import ZoomParticipantDetector, RealTimeVisualizer
from face_detector import FaceDetector
from notification_system import NotificationSystem, SoundNotification
from scheduler import ClassScheduler
from logger import AttendanceLogger

class CaptureThread(QThread):
    """
    실시간 화면 캡쳐 및 분석 스레드
    """
    
    # 시그널 정의
    frame_ready = pyqtSignal(np.ndarray)  # 시각화된 프레임 (UI 표시용)
    original_frame_ready = pyqtSignal(np.ndarray)  # 원본 프레임 (캡쳐 저장용)
    analysis_ready = pyqtSignal(int, int, list)  # 총 참가자, 얼굴 감지 수, 분석 결과
    error_occurred = pyqtSignal(str)
    
    def __init__(self, monitor_number: int = 2):
        """
        캡쳐 스레드 초기화
        
        Args:
            monitor_number (int): 모니터 번호
        """
        super().__init__()
        self.monitor_number = monitor_number
        self.running = False
        self.capture_interval = 5000  # 5초마다 캡쳐
        self.test_mode_active = False  # 테스트 모드 플래그
        
        # 모듈 초기화
        self.screen_capturer = ScreenCapture(monitor_number)
        self.zoom_detector = ZoomParticipantDetector()
        self.visualizer = RealTimeVisualizer()
        
        self.logger = logging.getLogger(__name__)
    
    def run(self):
        """
        스레드 실행
        """
        self.running = True

        while self.running:
            try:
                # 화면 캡쳐 (srcdc 오류 방지를 위한 추가 예외 처리)
                try:
                    screenshot = self.screen_capturer.capture_screen()
                except Exception as capture_error:
                    self.logger.warning(f"화면 캡쳐 일시 실패, 재시도: {capture_error}")
                    self.error_occurred.emit(f"화면 캡쳐 실패: {capture_error}")
                    self.msleep(500)  # 0.5초 대기 후 재시도
                    continue

                if screenshot is not None and screenshot.size > 0:
                    try:
                        # 테스트 모드일 때 강제 탐지 활성화
                        if self.test_mode_active:
                            # 강제로 얼굴 탐지 모델 로드
                            self.zoom_detector.face_detector._load_model()

                        # Zoom 참가자 분석
                        analysis_results, total_participants, face_detected = \
                            self.zoom_detector.detect_and_analyze_all(screenshot, force_detection=self.test_mode_active)

                        # 시각화 적용
                        visualized_frame = self.visualizer.draw_participant_boxes(
                            screenshot, analysis_results
                        )
                        visualized_frame = self.visualizer.draw_summary_info(
                            visualized_frame, total_participants, face_detected,
                            datetime.now().strftime("%H:%M:%S")
                        )

                        # 시그널 발송
                        self.frame_ready.emit(visualized_frame)  # UI 표시용 (시각화 포함)
                        self.original_frame_ready.emit(screenshot)  # 캡쳐 저장용 (원본)
                        self.analysis_ready.emit(total_participants, face_detected, analysis_results)

                    except Exception as analysis_error:
                        self.logger.error(f"분석 중 오류: {analysis_error}", exc_info=True)
                        self.error_occurred.emit(f"분석 오류: {analysis_error}")
                        # 분석 실패해도 원본 프레임은 표시
                        self.frame_ready.emit(screenshot)
                        self.original_frame_ready.emit(screenshot)

                # 지정된 간격만큼 대기
                self.msleep(self.capture_interval)

            except Exception as e:
                self.logger.error(f"캡쳐 스레드 오류: {e}", exc_info=True)
                self.error_occurred.emit(f"스레드 오류: {e}")
                self.msleep(5000)  # 오류 시 5초 대기
    
    def stop(self):
        """
        스레드 중지
        """
        self.running = False
        # 스크린 캡처 리소스 정리
        if hasattr(self, 'screen_capturer'):
            self.screen_capturer.cleanup()
        self.wait()
    
    def set_capture_interval(self, interval_ms: int):
        """
        캡쳐 간격 설정
        
        Args:
            interval_ms (int): 간격 (밀리초)
        """
        self.capture_interval = interval_ms
    
    def change_monitor(self, monitor_number: int):
        """
        모니터 변경
        
        Args:
            monitor_number (int): 새 모니터 번호
        """
        # 기존 리소스 정리
        if hasattr(self, 'screen_capturer'):
            self.screen_capturer.cleanup()
        
        self.monitor_number = monitor_number
        self.screen_capturer = ScreenCapture(monitor_number)

class ZoomAttendanceMainWindow(QMainWindow):
    """
    메인 윈도우 클래스
    """
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Zoom 강의 출석 자동화 v2.0")
        self.setGeometry(100, 100, 1200, 800)
        
        # 모듈 초기화
        self.monitor_manager = MonitorManager()
        self.notification_system = NotificationSystem()
        self.sound_notification = SoundNotification()
        self.attendance_logger = AttendanceLogger()
        
        # 스케줄러는 나중에 초기화
        self.scheduler = None
        self.capture_thread = None
        
        # 현재 상태 변수
        self.current_period = 0
        self.is_monitoring = False
        self.total_participants = 0
        self.face_detected_count = 0
        self.current_original_frame = None  # 캡쳐용 원본 프레임 저장
        
        # UI 라벨 초기화 (안전을 위한 기본값)
        self.status_labels = None
        
        # 교시별 캡처 관리
        self.period_capture_counts = {}  # {period: count} 각 교시별 캡처된 사진 수
        self.max_captures_per_period = 5  # 교시당 최대 캡처 수
        
        # 테스트 및 설정 변수
        self.test_detection_active = False
        self.manual_detection_timer = None
        self.settings = QSettings('ZoomAttendance', 'Settings')
        
        # 기본 설정값
        self.required_face_count = 1  # 필요한 최소 얼굴 수
        self.manual_duration = 30     # 수동 탐지 지속 시간 (초)
        self.class_schedules = {      # 교시별 활성화 설정
            1: True, 2: True, 3: True, 4: True,
            5: True, 6: True, 7: True, 8: True
        }
        
        # UI 초기화
        self.init_ui()
        self.init_system_tray()
        
        # 로깅 설정
        self.setup_logging()
        
        # 설정 로드
        self.load_settings()
        
        # UI 컨트롤에 설정값 반영
        self.update_ui_from_settings()
        
        # 모니터 자동 감지
        self.auto_detect_zoom_monitor()
        
        # 실시간 업데이트 타이머 시작
        self.start_realtime_updates()
    
    def init_ui(self):
        """
        UI 초기화 - 탭 기반 인터페이스
        """
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 메인 레이아웃
        main_layout = QVBoxLayout(central_widget)
        
        # 탭 위젯 생성
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # 탭 생성
        self.create_main_tab()      # 메인 모니터링
        self.create_settings_tab()  # 설정
    
    def create_main_tab(self):
        """
        메인 모니터링 탭 생성 - 실시간 미리보기와 상태 표시
        """
        main_tab = QWidget()
        self.tab_widget.addTab(main_tab, "📹 메인 모니터링")
        
        layout = QHBoxLayout(main_tab)
        
        # 왼쪽 패널 (상태 정보)
        left_panel = self.create_status_panel()
        layout.addWidget(left_panel, 1)
        
        # 오른쪽 패널 (실시간 미리보기)
        right_panel = self.create_preview_panel()
        layout.addWidget(right_panel, 2)
    
    def create_status_panel(self):
        """
        실시간 상태 정보 패널 생성
        """
        panel = QFrame()
        panel.setFrameStyle(QFrame.StyledPanel)
        layout = QVBoxLayout(panel)
        
        # 현재 시간 표시
        time_group = QGroupBox("📅 현재 시간")
        time_layout = QVBoxLayout(time_group)
        
        self.current_time_label = QLabel("--:--:--")
        self.current_time_label.setAlignment(Qt.AlignCenter)
        self.current_time_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #2196F3;")
        time_layout.addWidget(self.current_time_label)
        
        self.current_date_label = QLabel("----년 --월 --일")
        self.current_date_label.setAlignment(Qt.AlignCenter)
        self.current_date_label.setStyleSheet("font-size: 14px; color: #666;")
        time_layout.addWidget(self.current_date_label)
        
        layout.addWidget(time_group)
        
        # 현재 교시 표시
        class_group = QGroupBox("🎓 현재 교시")
        class_layout = QVBoxLayout(class_group)
        
        self.current_class_label = QLabel("수업 시간 아님")
        self.current_class_label.setAlignment(Qt.AlignCenter)
        self.current_class_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #FF5722;")
        class_layout.addWidget(self.current_class_label)
        
        layout.addWidget(class_group)
        
        # 감지 상태 표시
        detection_group = QGroupBox("👥 감지 상태")
        detection_layout = QVBoxLayout(detection_group)
        
        self.participant_count_label = QLabel("참여자: 0명")
        self.participant_count_label.setAlignment(Qt.AlignCenter)
        self.participant_count_label.setStyleSheet("font-size: 16px; color: #4CAF50;")
        detection_layout.addWidget(self.participant_count_label)
        
        self.face_count_label = QLabel("얼굴 감지: 0명")
        self.face_count_label.setAlignment(Qt.AlignCenter)
        self.face_count_label.setStyleSheet("font-size: 16px; color: #2196F3;")
        detection_layout.addWidget(self.face_count_label)
        
        layout.addWidget(detection_group)
        
        # 모니터링 상태
        monitoring_group = QGroupBox("🔍 모니터링 상태")
        monitoring_layout = QVBoxLayout(monitoring_group)
        
        self.monitoring_status_label = QLabel("❌ 중지됨")
        self.monitoring_status_label.setAlignment(Qt.AlignCenter)
        self.monitoring_status_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #F44336;")
        monitoring_layout.addWidget(self.monitoring_status_label)
        
        layout.addWidget(monitoring_group)
        
        # 스케줄 진행상황 (상세 정보)
        schedule_group = QGroupBox("📋 스케줄 진행상황")
        schedule_layout = QVBoxLayout(schedule_group)

        # 현재 교시 진행상황
        self.schedule_current_label = QLabel("대기 중...")
        self.schedule_current_label.setAlignment(Qt.AlignCenter)
        self.schedule_current_label.setStyleSheet("font-size: 13px; font-weight: bold; color: #2196F3;")
        self.schedule_current_label.setWordWrap(True)
        schedule_layout.addWidget(self.schedule_current_label)

        # 현재 시도 진행상황
        self.schedule_attempt_label = QLabel("")
        self.schedule_attempt_label.setAlignment(Qt.AlignCenter)
        self.schedule_attempt_label.setStyleSheet("font-size: 11px; color: #666;")
        self.schedule_attempt_label.setWordWrap(True)
        schedule_layout.addWidget(self.schedule_attempt_label)

        # 다음 시도 정보
        self.schedule_next_label = QLabel("")
        self.schedule_next_label.setAlignment(Qt.AlignCenter)
        self.schedule_next_label.setStyleSheet("font-size: 11px; color: #999;")
        self.schedule_next_label.setWordWrap(True)
        schedule_layout.addWidget(self.schedule_next_label)

        layout.addWidget(schedule_group)
        
        # 제어 버튼 섹션
        control_group = QGroupBox("🎮 제어")
        control_layout = QVBoxLayout(control_group)
        
        # 모니터링 시작/중지 버튼
        self.monitor_btn = QPushButton("모니터링 시작")
        self.monitor_btn.clicked.connect(self.toggle_monitoring)
        self.monitor_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-size: 14px; padding: 10px; }")
        control_layout.addWidget(self.monitor_btn)
        
        # 스케줄러 시작/중지 버튼
        self.scheduler_btn = QPushButton("자동 스케줄 시작")
        self.scheduler_btn.clicked.connect(self.toggle_scheduler)
        self.scheduler_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-size: 14px; padding: 10px; }")
        control_layout.addWidget(self.scheduler_btn)
        
        # 테스트 캡쳐 버튼
        test_btn = QPushButton("테스트 캡쳐")
        test_btn.clicked.connect(self.test_capture)
        test_btn.setStyleSheet("QPushButton { background-color: #FF9800; color: white; font-size: 12px; padding: 8px; }")
        control_layout.addWidget(test_btn)
        
        layout.addWidget(control_group)
        
        layout.addStretch()
        
        return panel
    
    def create_preview_panel(self):
        """
        실시간 미리보기 패널 생성
        """
        panel = QFrame()
        panel.setFrameStyle(QFrame.StyledPanel)
        layout = QVBoxLayout(panel)
        
        # 미리보기 화면
        preview_group = QGroupBox("📺 실시간 미리보기")
        preview_layout = QVBoxLayout(preview_group)
        
        self.preview_label = QLabel("모니터링을 시작하세요")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(640, 360)
        self.preview_label.setStyleSheet("border: 1px solid #ccc; background-color: #f5f5f5; color: #666;")
        preview_layout.addWidget(self.preview_label)
        
        layout.addWidget(preview_group)
        
        return panel
    
    def create_settings_tab(self):
        """
        설정 탭 생성 - 모니터 설정과 교시 설정 통합
        """
        settings_tab = QWidget()
        self.tab_widget.addTab(settings_tab, "⚙️ 설정")
        
        layout = QVBoxLayout(settings_tab)
        
        # 모니터 설정 그룹
        monitor_group = QGroupBox("📺 모니터 설정")
        monitor_layout = QVBoxLayout(monitor_group)
        
        # 모니터 콤보박스
        self.monitor_combo = QComboBox()
        self.update_monitor_list()
        monitor_layout.addWidget(QLabel("Zoom 실행 모니터 선택:"))
        monitor_layout.addWidget(self.monitor_combo)
        
        # 모니터 변경 버튼
        change_monitor_btn = QPushButton("🔄 모니터 변경")
        change_monitor_btn.clicked.connect(self.change_monitor)
        change_monitor_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-size: 12px; padding: 8px; }")
        monitor_layout.addWidget(change_monitor_btn)
        
        layout.addWidget(monitor_group)
        
        # 탐지 조건 설정 그룹
        detection_group = QGroupBox("👥 탐지 조건")
        detection_layout = QGridLayout(detection_group)
        
        detection_layout.addWidget(QLabel("수업 참여자 수 (강사포함):"), 0, 0)
        self.face_count_spinbox = QSpinBox()
        self.face_count_spinbox.setRange(1, 50)
        self.face_count_spinbox.setValue(self.required_face_count)
        self.face_count_spinbox.setSuffix("명")
        detection_layout.addWidget(self.face_count_spinbox, 0, 1)
        
        layout.addWidget(detection_group)
        
        # 교시별 설정 그룹
        schedule_group = QGroupBox("📅 교시별 자동 촬영 설정")
        schedule_layout = QGridLayout(schedule_group)
        
        # 전체 선택/해제 버튼
        select_all_layout = QHBoxLayout()
        select_all_btn = QPushButton("전체 선택")
        select_all_btn.clicked.connect(self.select_all_classes)
        deselect_all_btn = QPushButton("전체 해제")
        deselect_all_btn.clicked.connect(self.deselect_all_classes)
        
        select_all_layout.addWidget(select_all_btn)
        select_all_layout.addWidget(deselect_all_btn)
        select_all_layout.addStretch()
        
        layout.addLayout(select_all_layout)
        
        # 교시별 체크박스 생성 (수정된 시간표)
        self.class_checkboxes = {}
        class_times = [
            "09:30-10:30", "10:30-11:30", "11:30-12:30", "12:30-13:30",
            "14:30-15:30", "15:30-16:30", "16:30-17:30", "17:30-18:30"
        ]
        
        for i in range(8):
            period = i + 1
            time_text = class_times[i]
            
            # 4교시는 점심시간과 겹치므로 별도 표시
            if period == 4:
                checkbox = QCheckBox(f"{period}교시 ({time_text}) - 점심시간과 겹침")
            else:
                checkbox = QCheckBox(f"{period}교시 ({time_text})")
                
            checkbox.setChecked(self.class_schedules.get(period, True))
            
            self.class_checkboxes[period] = checkbox
            schedule_layout.addWidget(checkbox, i // 2, i % 2)
        
        # 점심시간 안내
        lunch_label = QLabel("🍽️ 점심시간 (13:30-14:30)은 자동으로 비활성화됩니다")
        lunch_label.setStyleSheet("QLabel { color: #888; font-style: italic; }")
        schedule_layout.addWidget(lunch_label, 4, 0, 1, 2)
        
        layout.addWidget(schedule_group)
        
        # 로그 섹션
        log_group = QGroupBox("📋 시스템 로그")
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(150)
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("font-family: 'Consolas', 'Monaco', monospace; font-size: 10px;")
        log_layout.addWidget(self.log_text)
        
        # 로그 제어 버튼
        log_btn_layout = QHBoxLayout()
        clear_log_btn = QPushButton("🗑️ 로그 지우기")
        clear_log_btn.clicked.connect(self.clear_log)
        clear_log_btn.setStyleSheet("QPushButton { background-color: #FF5722; color: white; font-size: 12px; padding: 5px; }")
        
        refresh_log_btn = QPushButton("🔄 로그 새로고침")
        refresh_log_btn.clicked.connect(self.refresh_log)
        refresh_log_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-size: 12px; padding: 5px; }")
        
        log_btn_layout.addWidget(clear_log_btn)
        log_btn_layout.addWidget(refresh_log_btn)
        log_btn_layout.addStretch()
        
        log_layout.addLayout(log_btn_layout)
        layout.addWidget(log_group)
        
        # 설정 저장 버튼
        save_settings_btn = QPushButton("💾 모든 설정 저장")
        save_settings_btn.clicked.connect(self.save_all_settings)
        save_settings_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-size: 14px; padding: 10px; }")
        layout.addWidget(save_settings_btn)
    
    def start_realtime_updates(self):
        """
        실시간 업데이트 타이머 시작
        """
        # 실시간 상태 업데이트 타이머 (1초마다)
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_realtime_status)
        self.status_timer.start(1000)  # 1초
        
        # 실시간 프리뷰 업데이트 타이머 (200ms마다)
        self.preview_timer = QTimer()
        self.preview_timer.timeout.connect(self.update_preview)
        self.preview_timer.start(200)  # 200ms
    
    def update_realtime_status(self):
        """
        실시간 상태 정보 업데이트
        """
        try:
            # 현재 시간 업데이트
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            current_date = now.strftime("%Y년 %m월 %d일")
            
            # 시간 라벨 업데이트 (안전 확인)
            if hasattr(self, 'current_time_label') and self.current_time_label:
                self.current_time_label.setText(current_time)
            if hasattr(self, 'current_date_label') and self.current_date_label:
                self.current_date_label.setText(current_date)
            
            # 현재 교시 확인 (기존 스케줄러 사용)
            if hasattr(self, 'scheduler') and self.scheduler:
                is_class, class_period = self.scheduler.is_class_time()
            else:
                # 스케줄러가 없으면 임시 확인용 스케줄러 사용 (로깅 없이)
                from scheduler import ClassScheduler
                temp_scheduler = ClassScheduler(capture_callback=None)
                is_class, class_period = temp_scheduler.is_class_time()
            
            # 교시 라벨 업데이트 (안전 확인)
            if hasattr(self, 'current_class_label') and self.current_class_label:
                if is_class:
                    self.current_class_label.setText(f"{class_period}교시 진행중")
                    self.current_class_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #4CAF50;")
                else:
                    self.current_class_label.setText("수업 시간 아님")
                    self.current_class_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #FF5722;")
            
            # 모니터링 상태 업데이트 (안전 확인)
            if hasattr(self, 'monitoring_status_label') and self.monitoring_status_label:
                if hasattr(self, 'capture_thread') and self.capture_thread and self.capture_thread.running:
                    self.monitoring_status_label.setText("✅ 모니터링 중")
                    self.monitoring_status_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #4CAF50;")
                else:
                    self.monitoring_status_label.setText("❌ 중지됨")
                    self.monitoring_status_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #F44336;")
            
            # 스케줄 진행상황 업데이트
            self.update_schedule_progress()

        except Exception as e:
            self.logger.error(f"실시간 상태 업데이트 오류: {e}")
    
    def update_schedule_progress(self):
        """
        스케줄 진행상황 업데이트 (상세 정보 표시)
        """
        try:
            if not hasattr(self, 'scheduler') or not self.scheduler:
                # 스케줄러가 없으면 대기 상태
                if hasattr(self, 'schedule_current_label'):
                    self.schedule_current_label.setText("자동 스케줄 대기 중")
                if hasattr(self, 'schedule_attempt_label'):
                    self.schedule_attempt_label.setText("")
                if hasattr(self, 'schedule_next_label'):
                    self.schedule_next_label.setText("")
                return

            from scheduler import ClassScheduler
            now = datetime.now()
            current_time = now.time()
            class_schedule = self.scheduler.class_schedule

            # 각 교시의 캡처 시간 확인 (35~40분, 5분간)
            for period, (start_time, end_time) in enumerate(class_schedule, 1):
                # 캡처 시작/종료 시간 계산
                capture_start_hour = start_time.hour
                capture_start_minute = start_time.minute + 35
                if capture_start_minute >= 60:
                    capture_start_hour += 1
                    capture_start_minute -= 60

                capture_end_hour = start_time.hour
                capture_end_minute = start_time.minute + 40
                if capture_end_minute >= 60:
                    capture_end_hour += 1
                    capture_end_minute -= 60

                from datetime import time
                capture_start = time(capture_start_hour, capture_start_minute)
                capture_end = time(capture_end_hour, capture_end_minute)

                # 현재 캡처 시간 중인 경우
                if capture_start <= current_time <= capture_end:
                    elapsed_minutes = (current_time.hour * 60 + current_time.minute) - \
                                    (capture_start_hour * 60 + capture_start_minute)
                    remaining_minutes = (capture_end_hour * 60 + capture_end_minute) - \
                                      (current_time.hour * 60 + current_time.minute)

                    # 현재 교시의 캡처 시도 횟수 확인
                    current_attempts = self.period_capture_counts.get(period, 0)
                    max_attempts = self.max_captures_per_period

                    if hasattr(self, 'schedule_current_label'):
                        self.schedule_current_label.setText(
                            f"📸 {period}교시 촬영 중 ({current_attempts}/{max_attempts}장)"
                        )

                    if hasattr(self, 'schedule_attempt_label'):
                        self.schedule_attempt_label.setText(
                            f"진행: {elapsed_minutes}분 경과 / {remaining_minutes}분 남음"
                        )

                    if hasattr(self, 'schedule_next_label'):
                        if current_attempts >= max_attempts:
                            self.schedule_next_label.setText(
                                f"✅ {period}교시 완료 (목표 달성)"
                            )
                        else:
                            self.schedule_next_label.setText(
                                f"다음 시도: 얼굴 감지 시 자동 촬영"
                            )
                    return

                # 다가오는 캡처 시간인 경우
                if current_time < capture_start:
                    time_until_start = (capture_start_hour * 60 + capture_start_minute) - \
                                     (current_time.hour * 60 + current_time.minute)

                    if hasattr(self, 'schedule_current_label'):
                        self.schedule_current_label.setText(
                            f"⏰ 다음: {period}교시 ({time_until_start}분 후)"
                        )

                    if hasattr(self, 'schedule_attempt_label'):
                        self.schedule_attempt_label.setText(
                            f"촬영 시작: {capture_start_hour:02d}:{capture_start_minute:02d}"
                        )

                    if hasattr(self, 'schedule_next_label'):
                        self.schedule_next_label.setText(
                            f"촬영 기간: 5분간 (목표 {self.max_captures_per_period}장)"
                        )
                    return

            # 오늘 모든 스케줄 종료
            if hasattr(self, 'schedule_current_label'):
                self.schedule_current_label.setText("📅 오늘 스케줄 종료")
            if hasattr(self, 'schedule_attempt_label'):
                total_captures = sum(self.period_capture_counts.values())
                self.schedule_attempt_label.setText(
                    f"총 {total_captures}장 촬영 완료"
                )
            if hasattr(self, 'schedule_next_label'):
                self.schedule_next_label.setText("내일 다시 시작됩니다")

        except Exception as e:
            self.logger.error(f"스케줄 진행상황 업데이트 오류: {e}")
            if hasattr(self, 'schedule_current_label'):
                self.schedule_current_label.setText("진행상황 확인 오류")

    def update_next_capture_time(self):
        """
        다음 자동 캡처 활성화 시간 업데이트
        """
        try:
            # 기존 스케줄러 사용 또는 임시 스케줄러 생성
            if hasattr(self, 'scheduler') and self.scheduler:
                class_schedule = self.scheduler.class_schedule
            else:
                from scheduler import ClassScheduler
                temp_scheduler = ClassScheduler(capture_callback=None)
                class_schedule = temp_scheduler.class_schedule
                
            now = datetime.now()
            current_time = now.time()
            
            # 각 교시의 35~40분 캡처 시간 확인 (5분간)
            for period, (start_time, end_time) in enumerate(class_schedule, 1):
                # 캡처 시작 시간 (교시 시작 + 35분)
                capture_start_hour = start_time.hour
                capture_start_minute = start_time.minute + 35
                
                if capture_start_minute >= 60:
                    capture_start_hour += 1
                    capture_start_minute -= 60
                
                # 캡처 종료 시간 (교시 시작 + 40분) - 5분간만
                capture_end_hour = start_time.hour
                capture_end_minute = start_time.minute + 40
                
                if capture_end_minute >= 60:
                    capture_end_hour += 1
                    capture_end_minute -= 60
                
                from datetime import time
                capture_start = time(capture_start_hour, capture_start_minute)
                capture_end = time(capture_end_hour, capture_end_minute)
                
                # 현재 시간이 이 캡처 시간보다 앞에 있으면
                if current_time < capture_start:
                    if hasattr(self, 'next_capture_label') and self.next_capture_label:
                        self.next_capture_label.setText(
                            f"다음 자동캡처 활성화\n{period}교시 {capture_start_hour:02d}:{capture_start_minute:02d}~{capture_end_hour:02d}:{capture_end_minute:02d}"
                        )
                    return
                
                # 현재 캡처 시간 중이면
                elif capture_start <= current_time <= capture_end:
                    remaining_minutes = (capture_end_hour * 60 + capture_end_minute) - (current_time.hour * 60 + current_time.minute)
                    if hasattr(self, 'next_capture_label') and self.next_capture_label:
                        self.next_capture_label.setText(
                            f"현재 자동캡처 활성화 중\n{period}교시 (종료까지 {remaining_minutes}분)"
                        )
                    return
            
            # 오늘 남은 캡처 시간이 없으면
            if hasattr(self, 'next_capture_label') and self.next_capture_label:
                self.next_capture_label.setText("오늘 예정된 자동캡처 없음")
            
        except Exception as e:
            if hasattr(self, 'next_capture_label') and self.next_capture_label:
                self.next_capture_label.setText("시간 계산 오류")
            self.logger.error(f"다음 캡처 시간 계산 오류: {e}")
    
    def update_preview(self):
        """
        실시간 미리보기 업데이트
        """
        try:
            if hasattr(self, 'capture_thread') and self.capture_thread and self.capture_thread.running:
                # 캡처 스레드가 실행 중이면 프레임 업데이트는 시그널로 처리
                pass
            else:
                # 모니터링이 중지된 상태면 기본 메시지 표시
                if not hasattr(self, '_preview_default_set'):
                    if hasattr(self, 'preview_label') and self.preview_label:
                        self.preview_label.setText("모니터링을 시작하세요")
                        self.preview_label.setStyleSheet("border: 1px solid #ccc; background-color: #f5f5f5; color: #666;")
                    self._preview_default_set = True
        except Exception as e:
            self.logger.error(f"미리보기 업데이트 오류: {e}")
    
    def toggle_main_monitoring(self):
        """
        메인 모니터링 및 자동스케줄 원버튼 토글
        """
        try:
            if hasattr(self, 'capture_thread') and self.capture_thread and self.capture_thread.running:
                # 현재 실행 중이면 중지
                self.stop_monitoring()
                self.main_monitoring_btn.setText("🚀 모니터링 & 자동스케줄 시작")
                self.main_monitoring_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-size: 16px; padding: 15px; font-weight: bold; }")
            else:
                # 중지 상태면 시작
                self.start_monitoring()
                # 자동 스케줄러도 함께 시작
                # TODO: 자동 스케줄러 시작 로직 추가
                self.main_monitoring_btn.setText("🛑 모니터링 & 자동스케줄 중지")
                self.main_monitoring_btn.setStyleSheet("QPushButton { background-color: #F44336; color: white; font-size: 16px; padding: 15px; font-weight: bold; }")
                
        except Exception as e:
            self.logger.error(f"메인 모니터링 토글 오류: {e}")
            QMessageBox.critical(self, "오류", f"모니터링 토글 중 오류가 발생했습니다:\n{e}")
    
    def save_all_settings(self):
        """
        모든 설정 저장 (교시 설정 + 기본 설정)
        """
        try:
            # 기본 설정 저장
            self.save_settings()
            # 교시 설정 저장
            self.save_schedule_settings()
            QMessageBox.information(self, "저장 완료", "모든 설정이 저장되었습니다.")
        except Exception as e:
            self.logger.error(f"설정 저장 오류: {e}")
            QMessageBox.critical(self, "오류", f"설정 저장 중 오류가 발생했습니다:\n{e}")
    
    def start_monitoring(self):
        """
        모니터링 시작
        """
        try:
            selected_monitor = self.monitor_combo.currentData() if hasattr(self, 'monitor_combo') else 2
            
            self.capture_thread = CaptureThread(selected_monitor)
            self.capture_thread.frame_ready.connect(self.update_screen)
            self.capture_thread.original_frame_ready.connect(self.store_original_frame)
            self.capture_thread.analysis_ready.connect(self.update_analysis)
            self.capture_thread.error_occurred.connect(self.handle_error)
            
            self.capture_thread.start()
            self.logger.info("실시간 모니터링 시작")
            
        except Exception as e:
            self.logger.error(f"모니터링 시작 오류: {e}")
            raise e
    
    def stop_monitoring(self):
        """
        모니터링 중지
        """
        try:
            if hasattr(self, 'capture_thread') and self.capture_thread:
                self.capture_thread.stop()
                self.capture_thread.wait()
                self.capture_thread = None
            
            # 미리보기 화면 초기화
            if hasattr(self, 'preview_label') and self.preview_label:
                self.preview_label.setText("모니터링을 시작하세요")
                self.preview_label.setStyleSheet("border: 1px solid #ccc; background-color: #f5f5f5; color: #666;")
                self._preview_default_set = True
            
            self.logger.info("실시간 모니터링 중지")
            
        except Exception as e:
            self.logger.error(f"모니터링 중지 오류: {e}")
    
    def clear_log(self):
        """
        GUI 로그 창 내용 지우기
        """
        if hasattr(self, 'log_text') and self.log_text:
            self.log_text.clear()
            self.logger.info("GUI 로그 창이 지워졌습니다")
    
    def refresh_log(self):
        """
        로그 파일에서 최근 로그를 다시 읽어와 표시
        """
        try:
            if hasattr(self, 'log_text') and self.log_text:
                # 현재 로그 창 내용 지우기
                self.log_text.clear()
                
                # 로그 파일에서 최근 50줄 읽기
                log_file_path = 'zoom_attendance_gui.log'
                if os.path.exists(log_file_path):
                    with open(log_file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        # 최근 50줄만 표시
                        recent_lines = lines[-50:] if len(lines) > 50 else lines
                        for line in recent_lines:
                            # 시간 포맷 조정
                            formatted_line = line.strip()
                            if ' - ' in formatted_line:
                                parts = formatted_line.split(' - ', 2)
                                if len(parts) >= 3:
                                    time_part = parts[0].split(' ')[1] if ' ' in parts[0] else parts[0]
                                    level_part = parts[1]
                                    msg_part = parts[2]
                                    formatted_line = f"[{time_part}] {level_part} - {msg_part}"
                            self.log_text.append(formatted_line)
                    
                    # 스크롤을 맨 아래로
                    cursor = self.log_text.textCursor()
                    cursor.movePosition(cursor.End)
                    self.log_text.setTextCursor(cursor)
                else:
                    self.log_text.append("[INFO] 로그 파일이 없습니다.")
                    
        except Exception as e:
            if hasattr(self, 'log_text') and self.log_text:
                self.log_text.append(f"[ERROR] 로그 새로고침 실패: {e}")
    
    def create_control_panel(self) -> QWidget:
        """
        컨트롤 패널 생성
        
        Returns:
            QWidget: 컨트롤 패널 위젯
        """
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 제목
        title_label = QLabel("출석 자동화 제어")
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # 모니터 선택 그룹
        monitor_group = QGroupBox("모니터 설정")
        monitor_layout = QVBoxLayout(monitor_group)
        
        # 모니터 콤보박스
        self.monitor_combo = QComboBox()
        self.update_monitor_list()
        monitor_layout.addWidget(QLabel("모니터 선택:"))
        monitor_layout.addWidget(self.monitor_combo)
        
        # 모니터 변경 버튼
        change_monitor_btn = QPushButton("모니터 변경")
        change_monitor_btn.clicked.connect(self.change_monitor)
        monitor_layout.addWidget(change_monitor_btn)
        
        layout.addWidget(monitor_group)
        
        # 제어 버튼 그룹
        control_group = QGroupBox("제어")
        control_layout = QVBoxLayout(control_group)
        
        # 모니터링 시작/중지 버튼
        self.monitor_btn = QPushButton("모니터링 시작")
        self.monitor_btn.clicked.connect(self.toggle_monitoring)
        self.monitor_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-size: 14px; padding: 10px; }")
        control_layout.addWidget(self.monitor_btn)
        
        # 스케줄러 시작/중지 버튼
        self.scheduler_btn = QPushButton("자동 스케줄 시작")
        self.scheduler_btn.clicked.connect(self.toggle_scheduler)
        self.scheduler_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-size: 14px; padding: 10px; }")
        control_layout.addWidget(self.scheduler_btn)
        
        # 테스트 캡쳐 버튼
        test_btn = QPushButton("테스트 캡쳐")
        test_btn.clicked.connect(self.test_capture)
        control_layout.addWidget(test_btn)
        
        layout.addWidget(control_group)
        
        # 상태 정보 그룹
        status_group = QGroupBox("현재 상태")
        status_layout = QGridLayout(status_group)
        
        # 상태 라벨들
        self.status_labels = {
            'period': QLabel("교시: -"),
            'participants': QLabel("참가자: 0명"),
            'detected': QLabel("얼굴 감지: 0명"),
            'rate': QLabel("감지율: 0%"),
            'monitor': QLabel("모니터: -"),
            'time': QLabel("시간: --:--:--")
        }
        
        row = 0
        for key, label in self.status_labels.items():
            label.setStyleSheet("QLabel { font-size: 12px; padding: 5px; }")
            status_layout.addWidget(label, row, 0)
            row += 1
        
        layout.addWidget(status_group)
        
        # 로그 텍스트
        log_group = QGroupBox("로그")
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(200)
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        
        layout.addWidget(log_group)
        
        # 하단 여백
        layout.addStretch()
        
        return panel
    
    def create_monitor_panel(self) -> QWidget:
        """
        모니터링 화면 패널 생성
        
        Returns:
            QWidget: 모니터링 패널 위젯
        """
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 제목
        title_label = QLabel("실시간 모니터링")
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # 화면 표시 라벨
        self.screen_label = QLabel()
        self.screen_label.setMinimumSize(640, 480)
        self.screen_label.setStyleSheet("QLabel { border: 2px solid #ddd; background-color: #f0f0f0; }")
        self.screen_label.setAlignment(Qt.AlignCenter)
        self.screen_label.setText("모니터링을 시작하세요")
        layout.addWidget(self.screen_label)
        
        # 하단 상태 인디케이터
        indicator_layout = QHBoxLayout()
        
        self.face_indicator = QLabel("얼굴 감지 상태")
        self.face_indicator.setStyleSheet("QLabel { background-color: #ff5555; color: white; padding: 10px; border-radius: 5px; }")
        self.face_indicator.setAlignment(Qt.AlignCenter)
        indicator_layout.addWidget(self.face_indicator)
        
        self.participant_indicator = QLabel("참가자: 0명")
        self.participant_indicator.setStyleSheet("QLabel { background-color: #555; color: white; padding: 10px; border-radius: 5px; }")
        self.participant_indicator.setAlignment(Qt.AlignCenter)
        indicator_layout.addWidget(self.participant_indicator)
        
        layout.addLayout(indicator_layout)
        
        return panel
    
    def init_system_tray(self):
        """
        시스템 트레이 초기화
        """
        if QSystemTrayIcon.isSystemTrayAvailable():
            self.tray_icon = QSystemTrayIcon(self)
            self.tray_icon.setIcon(QIcon.fromTheme("camera-video"))
            
            # 트레이 메뉴
            tray_menu = QMenu()
            
            show_action = QAction("창 보기", self)
            show_action.triggered.connect(self.show)
            tray_menu.addAction(show_action)
            
            hide_action = QAction("창 숨기기", self)
            hide_action.triggered.connect(self.hide)
            tray_menu.addAction(hide_action)
            
            tray_menu.addSeparator()
            
            quit_action = QAction("종료", self)
            quit_action.triggered.connect(self.close_application)
            tray_menu.addAction(quit_action)
            
            self.tray_icon.setContextMenu(tray_menu)
            self.tray_icon.show()
            
            self.tray_icon.messageClicked.connect(self.show)
    
    def setup_logging(self):
        """
        로깅 시스템 설정 (파일 + GUI 로깅)
        """
        # 기본 파일 로깅 설정
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('zoom_attendance_gui.log', encoding='utf-8'),
                logging.StreamHandler()  # 콘솔 출력
            ]
        )
        
        # GUI 로그 핸들러 추가 (log_text가 존재하는 경우에만)
        if hasattr(self, 'log_text') and self.log_text is not None:
            class GuiLogHandler(logging.Handler):
                def __init__(self, text_widget):
                    super().__init__()
                    self.text_widget = text_widget
                
                def emit(self, record):
                    try:
                        msg = self.format(record)
                        timestamp = datetime.now().strftime('%H:%M:%S')
                        self.text_widget.append(f"[{timestamp}] {msg}")
                        # 스크롤을 맨 아래로
                        cursor = self.text_widget.textCursor()
                        cursor.movePosition(cursor.End)
                        self.text_widget.setTextCursor(cursor)
                    except Exception:
                        pass  # GUI 오류 시 무시
            
            # GUI 핸들러 추가
            gui_handler = GuiLogHandler(self.log_text)
            gui_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(levelname)s - %(message)s')
            gui_handler.setFormatter(formatter)
            
            # 루트 로거에 추가
            root_logger = logging.getLogger()
            root_logger.addHandler(gui_handler)
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Zoom 출석 자동화 프로그램 시작")
    
    def update_monitor_list(self):
        """
        모니터 목록 업데이트
        """
        self.monitor_combo.clear()
        monitors = self.monitor_manager.list_all_monitors()
        
        for monitor in monitors:
            text = f"모니터 {monitor['number']} ({monitor['width']}x{monitor['height']})"
            self.monitor_combo.addItem(text, monitor['number'])
    
    def auto_detect_zoom_monitor(self):
        """
        Zoom 모니터 자동 감지
        """
        zoom_monitor = self.monitor_manager.find_zoom_monitor()
        
        # 콤보박스에서 해당 모니터 선택
        for i in range(self.monitor_combo.count()):
            if self.monitor_combo.itemData(i) == zoom_monitor:
                self.monitor_combo.setCurrentIndex(i)
                break
        
        if hasattr(self, 'status_labels') and self.status_labels:
            self.status_labels['monitor'].setText(f"모니터: {zoom_monitor}")
        self.logger.info(f"Zoom 모니터 자동 감지: 모니터 {zoom_monitor}")
    
    def change_monitor(self):
        """
        모니터 변경
        """
        selected_monitor = self.monitor_combo.currentData()
        
        if selected_monitor and self.capture_thread:
            self.capture_thread.change_monitor(selected_monitor)
            if hasattr(self, 'status_labels') and self.status_labels:
                self.status_labels['monitor'].setText(f"모니터: {selected_monitor}")
            self.notification_system.notify_monitor_switched(selected_monitor)
            self.logger.info(f"모니터 변경: {selected_monitor}")
    
    def toggle_monitoring(self):
        """
        모니터링 시작/중지
        """
        if not self.is_monitoring:
            # 모니터링 시작
            selected_monitor = self.monitor_combo.currentData() or 2
            
            self.capture_thread = CaptureThread(selected_monitor)
            self.capture_thread.frame_ready.connect(self.update_screen)
            self.capture_thread.original_frame_ready.connect(self.store_original_frame)
            self.capture_thread.analysis_ready.connect(self.update_analysis)
            self.capture_thread.error_occurred.connect(self.handle_error)
            
            self.capture_thread.start()
            
            self.is_monitoring = True
            self.monitor_btn.setText("모니터링 중지")
            self.monitor_btn.setStyleSheet("QPushButton { background-color: #f44336; color: white; font-size: 14px; padding: 10px; }")
            
            # 상태 업데이트 타이머
            self.status_timer = QTimer()
            self.status_timer.timeout.connect(self.update_status)
            self.status_timer.start(1000)  # 1초마다
            
            self.logger.info("실시간 모니터링 시작")
            
        else:
            # 모니터링 중지
            if self.capture_thread:
                self.capture_thread.stop()
                self.capture_thread = None
            
            if hasattr(self, 'status_timer'):
                self.status_timer.stop()
            
            self.is_monitoring = False
            self.monitor_btn.setText("모니터링 시작")
            self.monitor_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-size: 14px; padding: 10px; }")
            
            if hasattr(self, 'screen_label') and self.screen_label:
                self.screen_label.setText("모니터링을 시작하세요")
            if hasattr(self, 'face_indicator') and self.face_indicator:
                self.face_indicator.setText("얼굴 감지 상태")
                self.face_indicator.setStyleSheet("QLabel { background-color: #ff5555; color: white; padding: 10px; border-radius: 5px; }")
            
            self.logger.info("실시간 모니터링 중지")
    
    def toggle_scheduler(self):
        """
        스케줄러 시작/중지
        """
        if not self.scheduler:
            # 스케줄러 시작
            self.scheduler = ClassScheduler(capture_callback=self.scheduled_capture)
            
            try:
                # 별도 스레드에서 스케줄러 실행
                from threading import Thread
                self.scheduler_thread = Thread(target=self.scheduler.start, daemon=True)
                self.scheduler_thread.start()
                
                self.scheduler_btn.setText("자동 스케줄 중지")
                self.scheduler_btn.setStyleSheet("QPushButton { background-color: #ff9800; color: white; font-size: 14px; padding: 10px; }")
                
                self.notification_system.notify_system_start()
                self.logger.info("자동 스케줄러 시작")
                
            except Exception as e:
                self.logger.error(f"스케줄러 시작 실패: {e}")
                self.scheduler = None
        else:
            # 스케줄러 중지
            if self.scheduler:
                self.scheduler.stop()
                self.scheduler = None
            
            self.scheduler_btn.setText("자동 스케줄 시작")
            self.scheduler_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-size: 14px; padding: 10px; }")
            
            self.notification_system.notify_system_stop()
            self.logger.info("자동 스케줄러 중지")
    
    def scheduled_capture(self, period: int):
        """
        스케줄된 캡쳐 실행 (35-40분 시간대, 교시별 5장 제한)
        
        Args:
            period (int): 교시 번호
        """
        # 해당 교시의 캡처 시간인지 확인 (35-40분)
        if not self.is_capture_time_for_period(period):
            return
            
        # 해당 교시의 캡처 제한 확인
        if period in self.period_capture_counts:
            if self.period_capture_counts[period] >= self.max_captures_per_period:
                self.logger.info(f"{period}교시 캡처 완료 (5장 달성), 감지 중단")
                return
        else:
            self.period_capture_counts[period] = 0
            
        self.current_period = period
        self.logger.info(f"{period}교시 자동 캡쳐 시도 ({self.period_capture_counts[period] + 1}/5)")
        
        # 얼굴 감지 조건 확인 및 원본 프레임 저장 (모든 참가자가 감지된 경우만)
        if (self.current_original_frame is not None and 
            self.face_detected_count >= self.required_face_count and
            self.total_participants > 0 and
            self.face_detected_count == self.total_participants):
            
            # 원본 화면을 captures 폴더에 저장
            import os
            os.makedirs("captures", exist_ok=True)
            
            capture_count = self.period_capture_counts[period] + 1
            capture_filename = f"captures/{datetime.now().strftime('%Y%m%d')}_{period}교시_{capture_count}.png"
            cv2.imwrite(capture_filename, self.current_original_frame)
            
            # 캡처 카운트 증가
            self.period_capture_counts[period] += 1
            
            self.logger.info(f"출석 조건 만족 - 원본 화면 저장: {capture_filename} ({self.period_capture_counts[period]}/5)")
            self.attendance_logger.log_attendance(period, [capture_filename])
            self.notification_system.notify_capture_success(period, capture_filename)
        else:
            self.logger.info(f"{period}교시 - 출석 조건 미달 (감지: {self.face_detected_count}/{self.total_participants})")
        
        # GUI에서 교시 표시 업데이트
        if hasattr(self, 'status_labels') and self.status_labels:
            self.status_labels['period'].setText(f"교시: {period}")
    
    def is_capture_time_for_period(self, period: int) -> bool:
        """
        해당 교시의 캡처 시간인지 확인 (35-40분)
        
        Args:
            period (int): 교시 번호
            
        Returns:
            bool: 캡처 시간 여부
        """
        if hasattr(self, 'scheduler') and self.scheduler:
            current_time = datetime.now().time()
            
            # 해당 교시의 시간표 가져오기
            if period <= len(self.scheduler.class_schedule):
                start_time, end_time = self.scheduler.class_schedule[period - 1]
                
                # 캡처 시작 시간 (교시 시작 + 35분)
                capture_start_hour = start_time.hour
                capture_start_minute = start_time.minute + 35
                
                if capture_start_minute >= 60:
                    capture_start_hour += 1
                    capture_start_minute -= 60
                
                # 캡처 종료 시간 (교시 시작 + 40분)
                capture_end_hour = start_time.hour
                capture_end_minute = start_time.minute + 40
                
                if capture_end_minute >= 60:
                    capture_end_hour += 1
                    capture_end_minute -= 60
                
                from datetime import time
                capture_start = time(capture_start_hour, capture_start_minute)
                capture_end = time(capture_end_hour, capture_end_minute)
                
                return capture_start <= current_time <= capture_end
        
        return False
    
    def test_capture(self):
        """
        테스트 캡쳐 실행 (원본 프레임 사용)
        """
        try:
            if self.current_original_frame is not None:
                # 현재 저장된 원본 프레임 사용
                test_file = f"test_capture_{datetime.now().strftime('%H%M%S')}.png"
                cv2.imwrite(test_file, self.current_original_frame)
                
                self.logger.info(f"테스트 캡쳐 완료: {test_file} (원본 화면)")
                QMessageBox.information(self, "테스트 완료", f"테스트 캡쳐가 완료되었습니다.\n파일: {test_file}\n(시각화 효과 제외한 원본 화면)")
            else:
                # 모니터링이 실행되지 않은 경우 직접 캡쳐
                selected_monitor = self.monitor_combo.currentData() or 2
                capturer = ScreenCapture(selected_monitor)
                
                screenshot = capturer.capture_screen()
                if screenshot.size > 0:
                    test_file = f"test_capture_{datetime.now().strftime('%H%M%S')}.png"
                    cv2.imwrite(test_file, screenshot)
                    
                    self.logger.info(f"테스트 캡쳐 완료: {test_file}")
                    QMessageBox.information(self, "테스트 완료", f"테스트 캡쳐가 완료되었습니다.\n파일: {test_file}")
                else:
                    self.logger.error("테스트 캡쳐 실패")
                    QMessageBox.warning(self, "테스트 실패", "화면 캡쳐에 실패했습니다.")
                
        except Exception as e:
            self.logger.error(f"테스트 캡쳐 오류: {e}")
            QMessageBox.critical(self, "오류", f"테스트 중 오류가 발생했습니다:\n{e}")
    
    def update_screen(self, frame: np.ndarray):
        """
        화면 업데이트 - 메인 탭의 실시간 미리보기에 표시
        
        Args:
            frame (np.ndarray): 캡쳐된 프레임
        """
        try:
            # OpenCV BGR을 RGB로 변환
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # QImage로 변환
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            # 메인 탭의 미리보기 라벨 크기에 맞게 조정
            if hasattr(self, 'preview_label') and self.preview_label:
                label_size = self.preview_label.size()
                scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
                    label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                self.preview_label.setPixmap(scaled_pixmap)
                self._preview_default_set = False
            
            # 기존 screen_label도 업데이트 (호환성)
            if hasattr(self, 'screen_label') and self.screen_label:
                label_size = self.screen_label.size()
                scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
                    label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                self.screen_label.setPixmap(scaled_pixmap)
            
        except Exception as e:
            self.logger.error(f"화면 업데이트 오류: {e}")
    
    def store_original_frame(self, frame: np.ndarray):
        """
        원본 프레임 저장 (시각화 없는 버전)
        
        Args:
            frame (np.ndarray): 원본 캡쳐된 프레임
        """
        self.current_original_frame = frame.copy()
    
    def update_analysis(self, total_participants: int, face_detected: int, analysis_results: list):
        """
        분석 결과 업데이트 - 메인 탭 상태와 기존 상태 모두 업데이트
        
        Args:
            total_participants (int): 총 참가자 수
            face_detected (int): 얼굴 감지된 수
            analysis_results (list): 상세 분석 결과
        """
        self.total_participants = total_participants
        self.face_detected_count = face_detected
        
        # 메인 탭 상태 라벨 업데이트
        if hasattr(self, 'participant_count_label'):
            self.participant_count_label.setText(f"참여자: {total_participants}명")
        if hasattr(self, 'face_count_label'):
            self.face_count_label.setText(f"얼굴 감지: {face_detected}명")
        
        # 기존 상태 라벨 업데이트 (호환성)
        if hasattr(self, 'status_labels') and self.status_labels:
            self.status_labels['participants'].setText(f"참가자: {total_participants}명")
            self.status_labels['detected'].setText(f"얼굴 감지: {face_detected}명")
            
            if total_participants > 0:
                rate = (face_detected / total_participants) * 100
                self.status_labels['rate'].setText(f"감지율: {rate:.1f}%")
            else:
                self.status_labels['rate'].setText("감지율: 0%")
        
        # 인디케이터 업데이트
        self.participant_indicator.setText(f"참가자: {total_participants}명")
        
        # 필요한 최소 얼굴 수와 비교하여 상태 결정
        meets_requirement = face_detected >= self.required_face_count
        
        if meets_requirement and face_detected > 0:
            self.face_indicator.setText(f"✓ 출석 조건 만족 ({face_detected}/{self.required_face_count})")
            self.face_indicator.setStyleSheet("QLabel { background-color: #4CAF50; color: white; padding: 10px; border-radius: 5px; }")
        elif face_detected > 0:
            self.face_indicator.setText(f"⚠️ 부족 ({face_detected}/{self.required_face_count})")
            self.face_indicator.setStyleSheet("QLabel { background-color: #FF9800; color: white; padding: 10px; border-radius: 5px; }")
        else:
            self.face_indicator.setText("✗ 얼굴 없음")
            self.face_indicator.setStyleSheet("QLabel { background-color: #f44336; color: white; padding: 10px; border-radius: 5px; }")
    
    def update_status(self):
        """
        상태 정보 업데이트 (메인 탭과 컨트롤 탭 모두)
        """
        current_time = datetime.now().strftime("%H:%M:%S")
        
        # 컨트롤 탭의 status_labels 업데이트 (존재하는 경우)
        if hasattr(self, 'status_labels') and self.status_labels:
            self.status_labels['time'].setText(f"시간: {current_time}")
            
            # 현재 교시 확인
            if self.scheduler:
                is_class, period = self.scheduler.is_class_time()
                if is_class:
                    self.status_labels['period'].setText(f"교시: {period}")
                else:
                    self.status_labels['period'].setText("교시: 쉬는시간")
    
    def handle_error(self, error_message: str):
        """
        오류 처리
        
        Args:
            error_message (str): 오류 메시지
        """
        self.logger.error(f"캡쳐 오류: {error_message}")
        self.notification_system.notify_error(error_message)
    
    def closeEvent(self, event):
        """
        창 닫기 이벤트
        """
        # 시스템 트레이가 있으면 트레이로 최소화
        if hasattr(self, 'tray_icon') and self.tray_icon.isVisible():
            self.hide()
            self.tray_icon.showMessage(
                "출석 자동화",
                "프로그램이 시스템 트레이로 최소화되었습니다.",
                QSystemTrayIcon.Information,
                2000
            )
            event.ignore()
        else:
            self.close_application()
    
    def close_application(self):
        """
        애플리케이션 완전 종료
        """
        # 모니터링 중지
        if self.is_monitoring:
            self.toggle_monitoring()
        
        # 스케줄러 중지
        if self.scheduler:
            self.toggle_scheduler()
        
        # 시스템 트레이 제거
        if hasattr(self, 'tray_icon'):
            self.tray_icon.hide()
        
        self.logger.info("애플리케이션 종료")
        QApplication.quit()
    
    # === 새로운 기능들 ===
    
    def toggle_test_mode(self):
        """
        테스트 모드 온/오프 - 실시간 얼굴 탐지 시각화
        """
        self.test_detection_active = not self.test_detection_active
        
        if self.test_detection_active:
            # 테스트 모드 시작
            self.test_mode_btn.setText("🟢 테스트 모드 중지")
            self.test_mode_btn.setStyleSheet("QPushButton { background-color: #f44336; color: white; font-size: 14px; padding: 10px; }")
            self.logger.info("실시간 테스트 모드 시작 - 강제 얼굴 탐지 활성화")
            
            # 캡쳐 스레드가 실행 중이 아니면 시작
            if not self.is_monitoring:
                self.toggle_monitoring()
            
            # 캡쳐 스레드에 테스트 모드 설정
            if self.capture_thread:
                self.capture_thread.test_mode_active = True
                
        else:
            # 테스트 모드 중지
            self.test_mode_btn.setText("🔴 테스트 모드 시작")
            self.test_mode_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-size: 14px; padding: 10px; }")
            self.logger.info("실시간 테스트 모드 중지")
            
            # 캡쳐 스레드의 테스트 모드 해제
            if self.capture_thread:
                self.capture_thread.test_mode_active = False
    
    def start_manual_detection(self):
        """
        수동 탐지 시작 (지정된 시간 동안)
        """
        duration = self.duration_spinbox.value()
        
        if self.manual_detection_timer and self.manual_detection_timer.isActive():
            # 이미 실행 중이면 중지
            self.manual_detection_timer.stop()
            self.manual_detect_btn.setText("⏰ 지정 시간 탐지 시작")
            self.manual_detect_btn.setStyleSheet("QPushButton { background-color: #FF9800; color: white; font-size: 12px; padding: 8px; }")
            
            # 캡쳐 스레드의 테스트 모드 해제
            if self.capture_thread:
                self.capture_thread.test_mode_active = False
            
            self.logger.info("수동 탐지 중지")
            return
        
        # 캡쳐 스레드가 실행 중이 아니면 시작
        if not self.is_monitoring:
            self.toggle_monitoring()
        
        # 수동 탐지 시작
        self.manual_detect_btn.setText(f"⏹️ 탐지 중지 ({duration}초)")
        self.manual_detect_btn.setStyleSheet("QPushButton { background-color: #f44336; color: white; font-size: 12px; padding: 8px; }")
        
        # 캡쳐 스레드에 테스트 모드 설정
        if self.capture_thread:
            self.capture_thread.test_mode_active = True
        
        # 타이머 설정
        self.manual_detection_timer = QTimer()
        self.manual_detection_timer.setSingleShot(True)
        self.manual_detection_timer.timeout.connect(self.stop_manual_detection)
        self.manual_detection_timer.start(duration * 1000)  # 초를 밀리초로 변환
        
        self.logger.info(f"수동 탐지 시작: {duration}초간")
    
    def stop_manual_detection(self):
        """
        수동 탐지 중지
        """
        self.manual_detect_btn.setText("⏰ 지정 시간 탐지 시작")
        self.manual_detect_btn.setStyleSheet("QPushButton { background-color: #FF9800; color: white; font-size: 12px; padding: 8px; }")
        
        # 캡쳐 스레드의 테스트 모드 해제
        if self.capture_thread:
            self.capture_thread.test_mode_active = False
        
        self.logger.info("수동 탐지 완료")
    
    def save_settings(self):
        """
        설정 저장
        """
        # 현재 UI 값들을 변수에 저장
        self.required_face_count = self.face_count_spinbox.value()
        self.manual_duration = self.duration_spinbox.value()
        
        # QSettings에 저장
        self.settings.setValue('required_face_count', self.required_face_count)
        self.settings.setValue('manual_duration', self.manual_duration)
        
        self.logger.info(f"설정 저장됨: 최소 얼굴 수={self.required_face_count}, 수동 시간={self.manual_duration}초")
        
        # 사용자에게 알림
        QMessageBox.information(self, "설정 저장", 
                               f"설정이 저장되었습니다.\n\n"
                               f"• 최소 얼굴 수: {self.required_face_count}명\n"
                               f"• 수동 탐지 시간: {self.manual_duration}초")
    
    def save_schedule_settings(self):
        """
        교시별 스케줄 설정 저장
        """
        # 체크박스 상태를 딕셔너리에 저장
        for period, checkbox in self.class_checkboxes.items():
            self.class_schedules[period] = checkbox.isChecked()
        
        # QSettings에 저장
        self.settings.setValue('class_schedules', json.dumps(self.class_schedules))
        
        # 활성화된 교시 목록
        active_classes = [str(p) for p, active in self.class_schedules.items() if active]
        
        self.logger.info(f"교시 설정 저장됨: {', '.join(active_classes)}교시 활성화")
        
        # 사용자에게 알림
        QMessageBox.information(self, "교시 설정 저장", 
                               f"교시 설정이 저장되었습니다.\n\n"
                               f"활성화된 교시: {', '.join(active_classes)}교시")
    
    def select_all_classes(self):
        """
        모든 교시 선택
        """
        for checkbox in self.class_checkboxes.values():
            checkbox.setChecked(True)
    
    def deselect_all_classes(self):
        """
        모든 교시 선택 해제
        """
        for checkbox in self.class_checkboxes.values():
            checkbox.setChecked(False)
    
    def load_settings(self):
        """
        저장된 설정 로드
        """
        try:
            # 기본값 또는 저장된 값 로드
            self.required_face_count = int(self.settings.value('required_face_count', 1))
            self.manual_duration = int(self.settings.value('manual_duration', 30))
            
            # 교시 설정 로드
            saved_schedules = self.settings.value('class_schedules', None)
            if saved_schedules:
                self.class_schedules = json.loads(saved_schedules)
            
            self.logger.info("설정 로드 완료")
            
        except Exception as e:
            self.logger.error(f"설정 로드 실패: {e}")
            # 기본값 사용
            self.required_face_count = 1
            self.manual_duration = 30
            self.class_schedules = {i: True for i in range(1, 9)}
    
    def update_ui_from_settings(self):
        """
        설정값으로 UI 컨트롤 업데이트
        """
        try:
            # 스핀박스 값 설정
            if hasattr(self, 'face_count_spinbox'):
                self.face_count_spinbox.setValue(self.required_face_count)
            
            if hasattr(self, 'duration_spinbox'):
                self.duration_spinbox.setValue(self.manual_duration)
            
            # 교시별 체크박스 설정
            if hasattr(self, 'class_checkboxes'):
                for period, checkbox in self.class_checkboxes.items():
                    checkbox.setChecked(self.class_schedules.get(period, True))
                    
            self.logger.info("UI 설정값 반영 완료")
            
        except Exception as e:
            self.logger.error(f"UI 설정값 반영 실패: {e}")

def main():
    """
    메인 함수
    """
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)  # 트레이 모드 지원
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('zoom_attendance_gui.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    # 메인 윈도우 생성
    window = ZoomAttendanceMainWindow()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()