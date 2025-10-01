# 🚀 Zoom 출석 자동화 배포 가이드

## 📦 GitHub Actions로 Windows 실행파일 자동 빌드

### 1단계: GitHub 저장소 생성 및 업로드

```bash
# 1. Git 저장소 초기화
git init

# 2. 모든 파일 추가
git add .

# 3. 첫 번째 커밋
git commit -m "🎓 Zoom 출석 자동화 v2.0 - 초기 릴리즈

✨ 주요 기능:
- 듀얼 모니터 지원 (서브모니터 자동 감지)
- 실시간 얼굴 감지 및 시각화 (초록/빨간색)
- 자동 교시 스케줄링 (점심시간 12:30-14:30 제외)
- Zoom 참가자 박스 개별 인식
- 참가자 수 실시간 카운팅
- 교시별 알림 시스템
- CSV 로그 자동 기록

🖥️ 데스크톱 앱:
- PyQt5 GUI 인터페이스
- 시스템 트레이 지원
- 실시간 모니터링 화면

🔧 빌드 시스템:
- GitHub Actions 자동 빌드
- Windows용 단일 실행파일 (.exe)
- 배포용 ZIP 패키지 자동 생성"

# 4. GitHub에서 새 저장소 생성 (https://github.com/new)
# 저장소 이름: zoom-attendance-automation

# 5. 원격 저장소 연결
git remote add origin https://github.com/USERNAME/zoom-attendance-automation.git

# 6. 기본 브랜치 설정 및 푸시
git branch -M main
git push -u origin main
```

### 2단계: 자동 빌드 확인

1. **GitHub Actions 확인**
   - 저장소의 "Actions" 탭 클릭
   - "Build Zoom Attendance App" 워크플로우 실행 확인
   - 빌드 진행 상황 모니터링 (약 10-15분 소요)

2. **빌드 성공 시**
   - ✅ 초록색 체크마크 표시
   - "Releases" 탭에 새 릴리즈 자동 생성
   - Windows용 실행파일 다운로드 가능

3. **빌드 실패 시**
   - ❌ 빨간색 X 표시
   - 로그에서 오류 확인
   - 일반적인 해결 방법은 아래 참조

### 3단계: 배포 파일 다운로드

1. **릴리즈에서 다운로드**
   ```
   https://github.com/USERNAME/zoom-attendance-automation/releases
   ```

2. **다운로드 파일들**
   - `ZoomAttendance.exe` - 메인 실행파일
   - `ZoomAttendance_v2.0_Windows.zip` - 전체 패키지 (권장)

## 🔧 문제 해결

### GitHub Actions 빌드 오류

#### 오류 1: 의존성 설치 실패
```yaml
# requirements.txt에서 버전 충돌 시
# 해결: requirements.txt 수정
```

#### 오류 2: PyInstaller 실행 실패
- **원인**: Hidden imports 누락
- **해결**: `zoom_attendance.spec` 파일의 `hiddenimports` 목록 확인

#### 오류 3: 메모리 부족
- **원인**: TensorFlow 등 대용량 라이브러리
- **해결**: GitHub Actions는 자동으로 처리됨

### 실행파일 문제

#### 문제 1: 바이러스 백신 차단
```
해결법:
1. Windows Defender에서 파일 허용
2. 바이러스 백신 소프트웨어에서 예외 추가
3. 신뢰할 수 있는 게시자로 등록
```

#### 문제 2: 실행 시 오류
```
일반적인 해결법:
1. 관리자 권한으로 실행
2. Visual C++ Redistributable 설치
3. .NET Framework 최신 버전 설치
```

## 📋 배포 체크리스트

### 개발자용 (코드 업로드 전)
- [ ] 모든 기능 테스트 완료
- [ ] requirements.txt 업데이트
- [ ] README.md 내용 확인
- [ ] 버전 정보 업데이트
- [ ] 민감한 정보 제거 확인

### GitHub 설정
- [ ] 저장소 Public으로 설정 (또는 GitHub Pro)
- [ ] Actions 권한 활성화
- [ ] Releases 생성 권한 확인

### 사용자용 (다운로드 후)
- [ ] Windows 10/11 확인
- [ ] Python 설치 불필요 (단일 실행파일)
- [ ] 바이러스 백신 허용 설정
- [ ] 관리자 권한 실행 권장

## 🎯 배포 전략

### 베타 테스트
1. **내부 테스트**
   - 개발 환경에서 전체 기능 검증
   - 다양한 해상도/모니터 구성 테스트

2. **제한된 배포**
   - 소수 사용자에게 베타 버전 제공
   - 피드백 수집 및 버그 수정

3. **공개 릴리즈**
   - 안정 버전 GitHub Releases 게시
   - 사용 설명서 및 FAQ 제공

### 지속적 업데이트
```bash
# 새 기능 추가 후
git add .
git commit -m "✨ 새로운 기능: 설명"
git push origin main

# 자동으로 새 릴리즈 생성됨
```

## 📞 지원 및 문의

### 사용자 지원
- **GitHub Issues**: 버그 리포트 및 기능 요청
- **Wiki**: 자주 묻는 질문
- **Discussions**: 사용자 커뮤니티

### 개발자 정보
```yaml
프로젝트: Zoom 강의 출석 자동화 v2.0
라이선스: MIT (교육 목적)
기술 스택: Python, PyQt5, OpenCV, MTCNN
플랫폼: Windows 10/11
```

## 🔒 보안 고려사항

### 개인정보 보호
- 캡쳐된 이미지는 로컬에만 저장
- 얼굴 데이터는 감지 목적으로만 사용
- 정기적인 로그 정리 기능 제공

### 네트워크 보안
- 외부 서버 통신 없음
- 모든 처리는 로컬에서 수행
- 캡쳐 데이터 외부 전송 금지

---

## 🎉 성공적인 배포 완료!

GitHub Actions를 통해 자동으로 Windows용 실행파일이 생성되며, 사용자는 단순히 .exe 파일을 다운로드하여 바로 사용할 수 있습니다.

**다음 단계**: 사용자에게 GitHub Releases 링크 공유하여 최신 버전 다운로드 안내