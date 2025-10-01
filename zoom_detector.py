"""
Zoom 참가자 박스 감지 및 분석 모듈
화면에서 Zoom 참가자 박스들을 개별적으로 인식하고 얼굴 감지
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict
import logging
from face_detector import FaceDetector

class ZoomParticipantDetector:
    """
    Zoom 참가자 박스 감지 및 얼굴 인식 클래스
    메모리 효율적인 얼굴 감지기와 연동
    """
    
    def __init__(self, face_detector: FaceDetector = None):
        """
        Zoom 참가자 감지기 초기화
        
        Args:
            face_detector: 메모리 효율적인 얼굴 감지기 인스턴스
        """
        self.face_detector = face_detector or FaceDetector()
        self.logger = logging.getLogger(__name__)
        
        # 참가자 박스 감지 파라미터
        self.min_box_area = 5000      # 최소 박스 크기
        self.max_box_area = 200000    # 최대 박스 크기
        self.aspect_ratio_min = 0.5   # 최소 가로세로 비율
        self.aspect_ratio_max = 2.0   # 최대 가로세로 비율
    
    def detect_participant_boxes(self, image: np.ndarray) -> List[Dict]:
        """
        화면에서 Zoom 참가자 박스들을 감지
        
        Args:
            image (np.ndarray): 입력 이미지 (BGR)
            
        Returns:
            List[Dict]: 감지된 박스 정보 리스트
                       각 dict는 'bbox', 'area', 'center' 키를 포함
        """
        boxes = []
        
        try:
            # 그레이스케일 변환
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 가우시안 블러로 노이즈 제거
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # 적응형 임계값 적용
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # 모폴로지 연산으로 구조 정리
            kernel = np.ones((3, 3), np.uint8)
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            # 컨투어 찾기
            contours, _ = cv2.findContours(
                cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            for contour in contours:
                # 면적 계산
                area = cv2.contourArea(contour)
                
                if self.min_box_area < area < self.max_box_area:
                    # 바운딩 박스 계산
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # 가로세로 비율 확인
                    aspect_ratio = w / h if h > 0 else 0
                    
                    if self.aspect_ratio_min <= aspect_ratio <= self.aspect_ratio_max:
                        # 박스 중심점 계산
                        center_x = x + w // 2
                        center_y = y + h // 2
                        
                        box_info = {
                            'bbox': (x, y, w, h),
                            'area': area,
                            'center': (center_x, center_y),
                            'aspect_ratio': aspect_ratio
                        }
                        
                        boxes.append(box_info)
            
            # 면적 기준으로 정렬 (큰 것부터)
            boxes.sort(key=lambda x: x['area'], reverse=True)
            
            self.logger.debug(f"감지된 참가자 박스 수: {len(boxes)}")
            
        except Exception as e:
            self.logger.error(f"참가자 박스 감지 오류: {e}")
        
        return boxes
    
    def analyze_participant_box(self, image: np.ndarray, box: Dict, force_detection: bool = False) -> Dict:
        """
        개별 참가자 박스 분석 (얼굴 감지 포함)
        
        Args:
            image (np.ndarray): 전체 이미지
            box (Dict): 박스 정보
            force_detection (bool): 강제 탐지 모드
            
        Returns:
            Dict: 분석 결과
        """
        x, y, w, h = box['bbox']
        
        # 박스 영역 추출
        roi = image[y:y+h, x:x+w]
        
        analysis = {
            'bbox': box['bbox'],
            'has_face': False,
            'face_count': 0,
            'face_confidence': 0.0,
            'is_active': False,
            'brightness': 0.0
        }
        
        try:
            if roi.size > 0:
                # 얼굴 감지
                faces = self.face_detector.detect_faces(roi, force_detection=force_detection)
                
                if faces:
                    analysis['has_face'] = True
                    analysis['face_count'] = len(faces)
                    analysis['face_confidence'] = max([face['confidence'] for face in faces])
                
                # 밝기 분석 (활성 상태 판단)
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                analysis['brightness'] = np.mean(gray_roi)
                
                # 밝기가 일정 수준 이상이면 활성 상태로 간주
                analysis['is_active'] = analysis['brightness'] > 30
                
        except Exception as e:
            self.logger.error(f"참가자 박스 분석 오류: {e}")
        
        return analysis
    
    def detect_and_analyze_all(self, image: np.ndarray, force_detection: bool = False) -> Tuple[List[Dict], int, int]:
        """
        모든 참가자 박스를 감지하고 분석
        
        Args:
            image (np.ndarray): 입력 이미지
            force_detection (bool): 강제 탐지 모드
            
        Returns:
            Tuple[List[Dict], int, int]: (분석 결과 리스트, 총 참가자 수, 얼굴 감지된 수)
        """
        # 참가자 박스 감지
        boxes = self.detect_participant_boxes(image)
        
        # 각 박스 분석
        analysis_results = []
        face_detected_count = 0
        
        for box in boxes:
            analysis = self.analyze_participant_box(image, box, force_detection=force_detection)
            analysis_results.append(analysis)
            
            if analysis['has_face']:
                face_detected_count += 1
        
        total_participants = len(boxes)
        
        self.logger.info(f"총 참가자: {total_participants}, 얼굴 감지: {face_detected_count}")
        
        return analysis_results, total_participants, face_detected_count

class RealTimeVisualizer:
    """
    실시간 시각화 클래스
    얼굴 감지 상태를 화면에 표시
    """
    
    def __init__(self):
        """
        시각화 도구 초기화
        """
        self.logger = logging.getLogger(__name__)
        
        # 색상 정의
        self.color_face_detected = (0, 255, 0)      # 초록색 (얼굴 감지됨)
        self.color_no_face = (0, 0, 255)           # 빨간색 (얼굴 미감지)
        self.color_inactive = (128, 128, 128)       # 회색 (비활성)
        
        # 폰트 설정
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.font_thickness = 2
    
    def draw_participant_boxes(self, image: np.ndarray, analysis_results: List[Dict]) -> np.ndarray:
        """
        참가자 박스에 시각화 효과 추가
        
        Args:
            image (np.ndarray): 원본 이미지
            analysis_results (List[Dict]): 분석 결과
            
        Returns:
            np.ndarray: 시각화가 적용된 이미지
        """
        result_image = image.copy()
        
        for i, analysis in enumerate(analysis_results):
            x, y, w, h = analysis['bbox']
            
            # 박스 색상 결정
            if not analysis['is_active']:
                color = self.color_inactive
                status = "비활성"
            elif analysis['has_face']:
                color = self.color_face_detected
                status = f"얼굴감지 ({analysis['face_confidence']:.2f})"
            else:
                color = self.color_no_face
                status = "얼굴없음"
            
            # 박스 그리기
            cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 3)
            
            # 상태 텍스트 배경
            text_size = cv2.getTextSize(status, self.font, self.font_scale, self.font_thickness)[0]
            cv2.rectangle(result_image, 
                         (x, y - text_size[1] - 10), 
                         (x + text_size[0] + 10, y), 
                         color, -1)
            
            # 상태 텍스트
            cv2.putText(result_image, status, (x + 5, y - 5), 
                       self.font, self.font_scale, (255, 255, 255), self.font_thickness)
            
            # 참가자 번호
            cv2.putText(result_image, f"#{i+1}", (x + 5, y + 25), 
                       self.font, self.font_scale, color, self.font_thickness)
        
        return result_image
    
    def draw_summary_info(self, image: np.ndarray, total_participants: int, 
                         face_detected_count: int, current_time: str = "") -> np.ndarray:
        """
        요약 정보를 화면에 표시
        
        Args:
            image (np.ndarray): 이미지
            total_participants (int): 총 참가자 수
            face_detected_count (int): 얼굴 감지된 참가자 수
            current_time (str): 현재 시간
            
        Returns:
            np.ndarray: 정보가 추가된 이미지
        """
        result_image = image.copy()
        
        # 정보 박스 배경
        info_height = 120
        cv2.rectangle(result_image, (10, 10), (400, info_height), (0, 0, 0), -1)
        cv2.rectangle(result_image, (10, 10), (400, info_height), (255, 255, 255), 2)
        
        # 정보 텍스트들
        texts = [
            f"시간: {current_time}",
            f"총 참가자: {total_participants}명",
            f"얼굴 감지: {face_detected_count}명",
            f"감지율: {(face_detected_count/total_participants*100):.1f}%" if total_participants > 0 else "감지율: 0%"
        ]
        
        for i, text in enumerate(texts):
            y_pos = 35 + (i * 25)
            cv2.putText(result_image, text, (20, y_pos), 
                       self.font, self.font_scale, (255, 255, 255), self.font_thickness)
        
        return result_image
    
    def create_status_indicator(self, width: int = 300, height: int = 100, 
                              face_detected: bool = False, participant_count: int = 0) -> np.ndarray:
        """
        상태 표시기 생성 (별도 창용)
        
        Args:
            width (int): 너비
            height (int): 높이
            face_detected (bool): 얼굴 감지 여부
            participant_count (int): 참가자 수
            
        Returns:
            np.ndarray: 상태 표시기 이미지
        """
        # 배경 색상 결정
        if face_detected:
            bg_color = (0, 150, 0)  # 어두운 초록
            status_text = "출석 확인됨"
        else:
            bg_color = (0, 0, 150)  # 어두운 빨강
            status_text = "출석 미확인"
        
        # 이미지 생성
        status_img = np.full((height, width, 3), bg_color, dtype=np.uint8)
        
        # 텍스트 추가
        cv2.putText(status_img, status_text, (10, 30), 
                   self.font, 0.8, (255, 255, 255), 2)
        cv2.putText(status_img, f"참가자: {participant_count}명", (10, 65), 
                   self.font, 0.6, (255, 255, 255), 2)
        
        return status_img

# 테스트 코드
if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    
    # 감지기 생성
    detector = ZoomParticipantDetector()
    visualizer = RealTimeVisualizer()
    
    # 웹캠으로 테스트 (실제로는 화면 캡쳐 사용)
    cap = cv2.VideoCapture(0)
    
    if cap.isOpened():
        print("실시간 Zoom 참가자 감지 테스트 시작...")
        print("ESC 키를 눌러 종료")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 참가자 분석
            analysis_results, total, detected = detector.detect_and_analyze_all(frame)
            
            # 시각화
            result_frame = visualizer.draw_participant_boxes(frame, analysis_results)
            result_frame = visualizer.draw_summary_info(
                result_frame, total, detected, 
                current_time=cv2.getTickCount()
            )
            
            # 상태 표시기
            status_indicator = visualizer.create_status_indicator(
                face_detected=(detected > 0), participant_count=total
            )
            
            # 화면 표시
            cv2.imshow("Zoom 참가자 감지", result_frame)
            cv2.imshow("출석 상태", status_indicator)
            
            # ESC 키로 종료
            if cv2.waitKey(1) & 0xFF == 27:
                break
        
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("웹캠을 열 수 없습니다.")
    
    print("테스트 완료")