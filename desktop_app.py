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
    frame_ready = pyqtSignal(np.ndarray)
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
        self.capture_interval = 1000  # 1초마다 캡쳐
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
                # 화면 캡쳐
                screenshot = self.screen_capturer.capture_screen()
                
                if screenshot.size > 0:
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
                    self.frame_ready.emit(visualized_frame)
                    self.analysis_ready.emit(total_participants, face_detected, analysis_results)
                
                # 지정된 간격만큼 대기
                self.msleep(self.capture_interval)
                
            except Exception as e:
                self.error_occurred.emit(str(e))
                self.msleep(5000)  # 오류 시 5초 대기
    
    def stop(self):
        """
        스레드 중지
        """
        self.running = False
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
        self.create_test_tab()      # 테스트 및 설정
        self.create_schedule_tab()  # 교시 설정
    
    def create_main_tab(self):
        """
        메인 모니터링 탭 생성
        """
        main_tab = QWidget()
        self.tab_widget.addTab(main_tab, "📹 메인 모니터링")
        
        layout = QHBoxLayout(main_tab)
        
        # 왼쪽 패널 (컨트롤)
        left_panel = self.create_control_panel()
        layout.addWidget(left_panel, 1)
        
        # 오른쪽 패널 (모니터링 화면)
        right_panel = self.create_monitor_panel()
        layout.addWidget(right_panel, 2)
    
    def create_test_tab(self):
        """
        테스트 및 설정 탭 생성
        """
        test_tab = QWidget()
        self.tab_widget.addTab(test_tab, "🔧 테스트 & 설정")
        
        layout = QVBoxLayout(test_tab)
        
        # 테스트 섹션
        test_group = QGroupBox("실시간 테스트")
        test_layout = QVBoxLayout(test_group)
        
        # 테스트 모드 토글
        self.test_mode_btn = QPushButton("🔴 테스트 모드 시작")
        self.test_mode_btn.clicked.connect(self.toggle_test_mode)
        self.test_mode_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-size: 14px; padding: 10px; }")
        test_layout.addWidget(self.test_mode_btn)
        
        # 수동 탐지 섹션
        manual_group = QGroupBox("수동 탐지")
        manual_layout = QGridLayout(manual_group)
        
        # 지속 시간 설정
        manual_layout.addWidget(QLabel("탐지 시간 (초):"), 0, 0)
        self.duration_spinbox = QSpinBox()
        self.duration_spinbox.setRange(10, 300)
        self.duration_spinbox.setValue(self.manual_duration)
        self.duration_spinbox.setSuffix("초")
        manual_layout.addWidget(self.duration_spinbox, 0, 1)
        
        # 수동 탐지 시작 버튼
        self.manual_detect_btn = QPushButton("⏰ 지정 시간 탐지 시작")
        self.manual_detect_btn.clicked.connect(self.start_manual_detection)
        self.manual_detect_btn.setStyleSheet("QPushButton { background-color: #FF9800; color: white; font-size: 12px; padding: 8px; }")
        manual_layout.addWidget(self.manual_detect_btn, 1, 0, 1, 2)
        
        # 얼굴 수 설정 섹션
        face_group = QGroupBox("탐지 조건")
        face_layout = QGridLayout(face_group)
        
        face_layout.addWidget(QLabel("필요한 최소 얼굴 수:"), 0, 0)
        self.face_count_spinbox = QSpinBox()
        self.face_count_spinbox.setRange(1, 50)
        self.face_count_spinbox.setValue(self.required_face_count)
        self.face_count_spinbox.setSuffix("명")
        face_layout.addWidget(self.face_count_spinbox, 0, 1)
        
        # 설정 저장 버튼
        save_btn = QPushButton("💾 설정 저장")
        save_btn.clicked.connect(self.save_settings)
        save_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-size: 12px; padding: 8px; }")
        
        # 레이아웃 구성
        test_layout.addWidget(manual_group)
        test_layout.addWidget(face_group)
        layout.addWidget(test_group)
        layout.addWidget(save_btn)
        layout.addStretch()
    
    def create_schedule_tab(self):
        """
        교시별 스케줄 설정 탭 생성
        """
        schedule_tab = QWidget()
        self.tab_widget.addTab(schedule_tab, "📅 교시 설정")
        
        layout = QVBoxLayout(schedule_tab)
        
        # 교시별 설정 그룹
        schedule_group = QGroupBox("교시별 자동 촬영 설정")
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
        
        # 교시별 체크박스 생성
        self.class_checkboxes = {}
        class_times = [
            "09:30-10:20", "10:30-11:20", "11:30-12:20", "12:30-14:30 (점심)",
            "14:30-15:20", "15:30-16:20", "16:30-17:20", "17:30-18:20", "18:30-19:20"
        ]
        
        for i in range(8):
            period = i + 1
            if i == 3:  # 점심시간 건너뛰기
                continue
                
            time_text = class_times[i]
            checkbox = QCheckBox(f"{period}교시 ({time_text})")
            checkbox.setChecked(self.class_schedules.get(period, True))
            
            self.class_checkboxes[period] = checkbox
            schedule_layout.addWidget(checkbox, i // 2, i % 2)
        
        # 점심시간 비활성화 표시
        lunch_label = QLabel("🍽️ 점심시간 (12:30-14:30) - 자동 비활성화")
        lunch_label.setStyleSheet("QLabel { color: #888; font-style: italic; }")
        schedule_layout.addWidget(lunch_label, 4, 0, 1, 2)
        
        # 설정 저장 버튼
        save_schedule_btn = QPushButton("💾 교시 설정 저장")
        save_schedule_btn.clicked.connect(self.save_schedule_settings)
        save_schedule_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-size: 14px; padding: 10px; }")
        
        layout.addWidget(schedule_group)
        layout.addWidget(save_schedule_btn)
        layout.addStretch()
    
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
        로깅 시스템 설정
        """
        # GUI 로그 핸들러 생성
        class GuiLogHandler(logging.Handler):
            def __init__(self, text_widget):
                super().__init__()
                self.text_widget = text_widget
            
            def emit(self, record):
                msg = self.format(record)
                self.text_widget.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
                # 스크롤을 맨 아래로
                self.text_widget.moveCursor(self.text_widget.textCursor().End)
        
        # 핸들러 추가
        gui_handler = GuiLogHandler(self.log_text)
        gui_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(levelname)s - %(message)s')
        gui_handler.setFormatter(formatter)
        
        # 루트 로거에 추가
        logging.getLogger().addHandler(gui_handler)
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Zoom 출석 자동화 시스템 시작")
    
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
        
        self.status_labels['monitor'].setText(f"모니터: {zoom_monitor}")
        self.logger.info(f"Zoom 모니터 자동 감지: 모니터 {zoom_monitor}")
    
    def change_monitor(self):
        """
        모니터 변경
        """
        selected_monitor = self.monitor_combo.currentData()
        
        if selected_monitor and self.capture_thread:
            self.capture_thread.change_monitor(selected_monitor)
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
            
            self.screen_label.setText("모니터링을 시작하세요")
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
        스케줄된 캡쳐 실행
        
        Args:
            period (int): 교시 번호
        """
        self.current_period = period
        self.notification_system.notify_capture_start(period)
        self.logger.info(f"{period}교시 자동 캡쳐 시작")
        
        # GUI에서 교시 표시 업데이트
        self.status_labels['period'].setText(f"교시: {period}")
    
    def test_capture(self):
        """
        테스트 캡쳐 실행
        """
        try:
            selected_monitor = self.monitor_combo.currentData() or 2
            capturer = ScreenCapture(selected_monitor)
            
            screenshot = capturer.capture_screen()
            if screenshot.size > 0:
                # 테스트 이미지 저장
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
        화면 업데이트
        
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
            
            # QLabel 크기에 맞게 조정
            label_size = self.screen_label.size()
            scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
                label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            
            self.screen_label.setPixmap(scaled_pixmap)
            
        except Exception as e:
            self.logger.error(f"화면 업데이트 오류: {e}")
    
    def update_analysis(self, total_participants: int, face_detected: int, analysis_results: list):
        """
        분석 결과 업데이트
        
        Args:
            total_participants (int): 총 참가자 수
            face_detected (int): 얼굴 감지된 수
            analysis_results (list): 상세 분석 결과
        """
        self.total_participants = total_participants
        self.face_detected_count = face_detected
        
        # 상태 라벨 업데이트
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
        상태 정보 업데이트
        """
        current_time = datetime.now().strftime("%H:%M:%S")
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