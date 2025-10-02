#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
단일 실행파일 빌드 스크립트
PyInstaller를 사용하여 배포용 실행파일 생성
"""

import os
import sys
import subprocess
import shutil
import platform
import locale
from pathlib import Path

# Windows에서 UTF-8 인코딩 설정
if platform.system() == "Windows":
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

class ZoomAttendanceBuilder:
    """
    Zoom 출석 자동화 프로그램 빌드 도구
    """
    
    def __init__(self):
        self.system = platform.system()
        self.project_dir = Path(__file__).parent
        self.dist_dir = self.project_dir / "dist"
        self.build_dir = self.project_dir / "build"
        
        print(f"[BUILD] Build System: {self.system}")
        print(f"[PROJECT] Project Path: {self.project_dir}")
    
    def clean_build_dirs(self):
        """
        이전 빌드 결과물 정리
        """
        print("\n[CLEAN] Cleaning previous build artifacts...")
        
        dirs_to_clean = [self.dist_dir, self.build_dir]
        
        for dir_path in dirs_to_clean:
            try:
                if dir_path.exists():
                    shutil.rmtree(dir_path)
                    print(f"  [OK] Removed {dir_path}")
                else:
                    print(f"  [SKIP] {dir_path} does not exist")
            except Exception as e:
                print(f"  [WARN] Failed to remove {dir_path}: {e}")
                # Windows에서 권한 문제로 삭제 실패해도 계속 진행
                continue
        
        print("  [SUCCESS] Build cleanup completed")
        return True
    
    def check_dependencies(self):
        """
        필요한 의존성 확인
        """
        print("\n[DEPS] Checking dependencies...")
        
        # 패키지명과 실제 import명이 다른 경우를 매핑
        package_mapping = {
            'PyInstaller': 'PyInstaller',
            'opencv-python': 'cv2',
            'mtcnn': 'mtcnn', 
            'mss': 'mss',
            'PyQt5': 'PyQt5',
            'tensorflow': 'tensorflow'
        }
        
        missing_packages = []
        
        for package_name, import_name in package_mapping.items():
            try:
                __import__(import_name)
                print(f"  [OK] {package_name}")
            except ImportError:
                missing_packages.append(package_name)
                print(f"  [ERROR] {package_name} - 누락")
        
        if missing_packages:
            print(f"\n[WARN]  누락된 패키지: {', '.join(missing_packages)}")
            print("다음 명령으로 설치하세요:")
            print(f"pip install {' '.join(missing_packages)}")
            return False
        
        print("[SUCCESS] 모든 의존성 확인 완료")
        return True
    
    def create_icon(self):
        """
        기본 아이콘 생성
        """
        print("\n[ICON] 아이콘 생성 중...")
        
        try:
            from PIL import Image, ImageDraw
            
            # 간단한 아이콘 생성
            size = 256
            img = Image.new('RGBA', (size, size), (0, 100, 200, 255))
            draw = ImageDraw.Draw(img)
            
            # 카메라 모양 그리기
            margin = 40
            draw.rounded_rectangle(
                [margin, margin + 30, size - margin, size - margin - 20],
                radius=20,
                fill=(255, 255, 255, 255),
                outline=(0, 0, 0, 255),
                width=5
            )
            
            # 렌즈 그리기
            center = size // 2
            lens_radius = 50
            draw.ellipse(
                [center - lens_radius, center - lens_radius + 15, 
                 center + lens_radius, center + lens_radius + 15],
                fill=(0, 0, 0, 255)
            )
            
            # 내부 렌즈
            inner_radius = 30
            draw.ellipse(
                [center - inner_radius, center - inner_radius + 15,
                 center + inner_radius, center + inner_radius + 15],
                fill=(100, 100, 100, 255)
            )
            
            # 아이콘 저장
            assets_dir = self.project_dir / "assets"
            assets_dir.mkdir(exist_ok=True)
            
            # PNG 형식으로 저장
            icon_png = assets_dir / "icon.png"
            img.save(icon_png, "PNG")
            
            # ICO 형식으로 저장 (Windows용)
            if self.system == "Windows":
                icon_ico = assets_dir / "icon.ico"
                img.save(icon_ico, "ICO", sizes=[(256, 256), (128, 128), (64, 64), (32, 32), (16, 16)])
                print(f"  [OK] Windows 아이콘 생성: {icon_ico}")
            
            print(f"  [OK] PNG 아이콘 생성: {icon_png}")
            return True
            
        except Exception as e:
            print(f"  [WARN]  아이콘 생성 실패: {e}")
            print("  기본 아이콘 없이 빌드 진행")
            return False
    
    def update_spec_file(self):
        """
        .spec 파일 생성 또는 업데이트
        """
        print("\n[SPEC] spec 파일 처리 중...")
        
        spec_file = self.project_dir / "zoom_attendance.spec"
        
        if not spec_file.exists():
            print("  [INFO] spec 파일이 없습니다. 새로 생성합니다.")
            return self._create_spec_file()
        else:
            print("  [INFO] 기존 spec 파일을 업데이트합니다.")
            return self._update_existing_spec_file()
    
    def _create_spec_file(self):
        """
        새로운 spec 파일 생성
        """
        # 아이콘 경로 설정
        assets_dir = self.project_dir / "assets"
        if self.system == "Windows":
            icon_path = assets_dir / "icon.ico"
        else:
            icon_path = assets_dir / "icon.png"
        
        icon_line = f"icon='{icon_path}'," if icon_path.exists() else "# icon=None,"
        
        spec_content = f"""# -*- mode: python ; coding: utf-8 -*-

a = Analysis(
    ['desktop_app.py'],
    pathex=['{self.project_dir}'],
    binaries=[],
    datas=[],
    hiddenimports=[
        'PyQt5.QtCore',
        'PyQt5.QtWidgets', 
        'PyQt5.QtGui',
        'cv2',
        'mtcnn',
        'tensorflow',
        'mss',
        'numpy',
        'pandas'
    ],
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='ZoomAttendance',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    {icon_line}
)
"""
        
        spec_file = self.project_dir / "zoom_attendance.spec"
        with open(spec_file, 'w', encoding='utf-8') as f:
            f.write(spec_content)
        
        print(f"  [OK] 새 spec 파일 생성 완료")
        return True
    
    def _update_existing_spec_file(self):
        """
        기존 spec 파일 업데이트
        """
        spec_file = self.project_dir / "zoom_attendance.spec"
        
        # spec 파일 읽기
        with open(spec_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 아이콘 경로 업데이트
        assets_dir = self.project_dir / "assets"
        if self.system == "Windows":
            icon_path = assets_dir / "icon.ico"
        else:
            icon_path = assets_dir / "icon.png"
        
        if icon_path.exists():
            # 기존 아이콘 라인을 새로운 경로로 교체
            content = content.replace(
                "# icon='assets/icon.ico',",
                f"icon='{icon_path}'"
            ).replace(
                "icon='assets/icon.ico',",
                f"icon='{icon_path}'"
            )
        
        # 업데이트된 내용 저장
        with open(spec_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"  [OK] spec 파일 업데이트 완료")
        return True
    
    def build_executable(self):
        """
        실행파일 빌드
        """
        print("\n[BUILD]  실행파일 빌드 중...")
        print("이 과정은 몇 분이 소요될 수 있습니다...")
        
        # PyInstaller 명령 실행
        spec_file = self.project_dir / "zoom_attendance.spec"
        
        cmd = [
            sys.executable, "-m", "PyInstaller",
            "--clean",
            str(spec_file)
        ]
        
        try:
            # 빌드 실행
            result = subprocess.run(
                cmd,
                cwd=self.project_dir,
                capture_output=True,
                text=True,
                timeout=1800  # 30분 타임아웃
            )
            
            if result.returncode == 0:
                print("  [SUCCESS] 빌드 성공!")
                return True
            else:
                print("  [ERROR] 빌드 실패!")
                print("오류 출력:")
                print(result.stderr)
                return False
                
        except subprocess.TimeoutExpired:
            print("  [TIMEOUT] 빌드 타임아웃 (30분 초과)")
            return False
        except Exception as e:
            print(f"  [ERROR] 빌드 중 예외 발생: {e}")
            return False
    
    def create_distribution_package(self):
        """
        배포 패키지 생성
        """
        print("\n[PACKAGE] 배포 패키지 생성 중...")
        
        # 실행파일 확인
        if self.system == "Windows":
            exe_name = "ZoomAttendance.exe"
        else:
            exe_name = "ZoomAttendance"
        
        exe_path = self.dist_dir / exe_name
        
        if not exe_path.exists():
            print(f"  [ERROR] 실행파일을 찾을 수 없습니다: {exe_path}")
            return False
        
        # 배포 디렉토리 생성
        release_dir = self.project_dir / "release"
        release_dir.mkdir(exist_ok=True)
        
        # 배포 패키지 이름
        version = "v2.0"
        package_name = f"ZoomAttendance_{version}_{self.system}"
        package_dir = release_dir / package_name
        
        # 기존 패키지 삭제
        if package_dir.exists():
            shutil.rmtree(package_dir)
        
        package_dir.mkdir()
        
        # 실행파일 복사
        shutil.copy2(exe_path, package_dir / exe_name)
        print(f"  [OK] 실행파일 복사: {exe_name}")
        
        # 필수 파일들 복사
        essential_files = [
            "README.md",
            "attendance_log.csv.example"  # 예시 파일
        ]
        
        for file_name in essential_files:
            src_file = self.project_dir / file_name
            if src_file.exists():
                shutil.copy2(src_file, package_dir / file_name)
                print(f"  [OK] 파일 복사: {file_name}")
        
        # README.md가 없으면 생성
        readme_path = package_dir / "README.md"
        if not readme_path.exists():
            self._create_distribution_readme(readme_path)
        
        # 실행 스크립트 생성
        self._create_run_scripts(package_dir, exe_name)
        
        # ZIP 파일 생성
        zip_path = release_dir / f"{package_name}.zip"
        shutil.make_archive(str(zip_path).replace('.zip', ''), 'zip', package_dir)
        
        print(f"  [SUCCESS] 배포 패키지 생성 완료!")
        print(f"  [FOLDER] 패키지 위치: {package_dir}")
        print(f"  [PACKAGE] ZIP 파일: {zip_path}")
        
        return True
    
    def _create_distribution_readme(self, readme_path: Path):
        """
        배포용 README 생성
        """
        content = """# Zoom 강의 출석 자동화 v2.0

## [SUCCESS] 사용법

### Windows
1. `ZoomAttendance.exe`를 더블클릭하여 실행
2. 또는 `실행.bat` 파일 사용

### Mac/Linux  
1. 터미널에서 `./ZoomAttendance` 실행
2. 또는 `실행.sh` 스크립트 사용

## [FEATURES] 기능

- [SUCCESS] 실시간 얼굴 감지 및 시각화
- [SUCCESS] 듀얼 모니터 지원
- [SUCCESS] 자동 스케줄링 (각 교시 35~45분)
- [SUCCESS] 알림 시스템
- [SUCCESS] CSV 로그 기록

## [SETTINGS] 설정

1. **모니터 선택**: 프로그램에서 Zoom이 실행 중인 모니터 선택
2. **모니터링 시작**: 실시간 화면 감지 활성화  
3. **자동 스케줄**: 교시별 자동 출석 체크 활성화

## [FILES] 생성되는 파일

- `captures/`: 캡쳐된 이미지 저장 폴더
- `attendance_log.csv`: 출석 기록 로그
- `zoom_attendance_gui.log`: 프로그램 실행 로그

## [TROUBLESHOOT] 문제 해결

### 화면 캡쳐 권한 (macOS)
1. 시스템 환경설정 > 보안 및 개인정보 보호
2. 개인정보 보호 > 화면 기록  
3. ZoomAttendance 앱 권한 허용

### 얼굴 감지 정확도
- Zoom 참가자 화면이 충분히 밝고 선명한지 확인
- 카메라가 정면을 향하고 있는지 확인

## 📞 지원

문제가 발생하면 로그 파일(`zoom_attendance_gui.log`)을 확인하세요.

---
© 2024 Zoom 출석 자동화 시스템
"""
        
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def _create_run_scripts(self, package_dir: Path, exe_name: str):
        """
        실행 스크립트 생성
        """
        if self.system == "Windows":
            # Windows 배치 파일
            bat_content = f"""@echo off
echo Zoom 출석 자동화 시스템 시작...
echo.
"{exe_name}"
if errorlevel 1 (
    echo.
    echo 프로그램 실행 중 오류가 발생했습니다.
    echo 관리자 권한으로 다시 시도해보세요.
    pause
)
"""
            with open(package_dir / "실행.bat", 'w', encoding='utf-8') as f:
                f.write(bat_content)
            
            print("  [OK] Windows 실행 스크립트 생성: 실행.bat")
        
        else:
            # Unix/Mac 셸 스크립트
            sh_content = f"""#!/bin/bash
echo "Zoom 출석 자동화 시스템 시작..."
echo

# 실행 권한 확인
if [ ! -x "./{exe_name}" ]; then
    echo "실행 권한 설정 중..."
    chmod +x "./{exe_name}"
fi

# 프로그램 실행
"./{exe_name}"

echo
echo "프로그램이 종료되었습니다."
read -p "Enter 키를 눌러 닫기..."
"""
            script_path = package_dir / "실행.sh"
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(sh_content)
            
            # 실행 권한 부여
            os.chmod(script_path, 0o755)
            
            print("  [OK] Unix/Mac 실행 스크립트 생성: 실행.sh")
    
    def build(self):
        """
        전체 빌드 프로세스 실행
        """
        print("[FACTORY] Zoom 출석 자동화 빌드 시작")
        print("=" * 50)
        
        steps = [
            ("의존성 확인", self.check_dependencies),
            ("이전 빌드 정리", self.clean_build_dirs),
            ("아이콘 생성", self.create_icon),
            ("spec 파일 업데이트", self.update_spec_file),
            ("실행파일 빌드", self.build_executable),
            ("배포 패키지 생성", self.create_distribution_package)
        ]
        
        for step_name, step_func in steps:
            print(f"\n[단계] {step_name}")
            print("-" * 30)
            
            try:
                if not step_func():
                    print(f"\n[ERROR] 빌드 실패: {step_name}")
                    return False
            except Exception as e:
                print(f"\n[ERROR] 빌드 오류: {step_name}")
                print(f"상세 오류: {e}")
                return False
        
        print("\n" + "=" * 50)
        print("[COMPLETE] 빌드 완료!")
        print("\n[FOLDER] 생성된 파일:")
        print(f"  - 실행파일: dist/")
        print(f"  - 배포 패키지: release/")
        print("\n[SUCCESS] 배포 준비 완료!")
        
        return True

def main():
    """
    메인 함수
    """
    builder = ZoomAttendanceBuilder()
    
    try:
        success = builder.build()
        if success:
            print("\n[SUCCESS] 빌드가 성공적으로 완료되었습니다!")
            sys.exit(0)
        else:
            print("\n[ERROR] 빌드 실패")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n[STOP]  사용자에 의해 빌드가 취소되었습니다.")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] 예상치 못한 오류: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()