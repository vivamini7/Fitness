import cv2
import time
import numpy as np

from angle_calculator import AngleCalculator
from dtw_algorithm import DTWMatcher  
from score_calculator import ScoreCalculator

class HardDemoMain:
    """Hard Demo 메인 통합 시스템"""
    
    def __init__(self, reference_csv_path="candy_angles.csv"):
        print("🎵 Hard Demo 시스템 초기화 중... 🎵")
        
        # 핵심 모듈들 초기화
        try:
            self.angle_calculator = AngleCalculator()
            self.dtw_matcher = DTWMatcher(reference_csv_path)
            self.score_calculator = ScoreCalculator()
            print("✅ 모든 모듈 초기화 완료")
        except Exception as e:
            print(f"❌ 모듈 초기화 실패: {e}")
            raise
        
        # 게임 상태 관리
        self.game_state = 'waiting'  # waiting, countdown, recording, processing, result
        self.countdown_duration = 3.0  # 3초 카운트다운
        self.recording_duration = 22.0  # 22초 녹화
        
        # 타이밍 관리
        self.countdown_start_time = None
        self.recording_start_time = None
        
        # 결과 저장
        self.final_score = None
        self.final_grade = None
        
        # 웹캠 설정
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("🎮 Hard Demo 준비 완료!")
        self.print_controls()
    
    def print_controls(self):
        """조작법 출력"""
        print("\n🎮 조작법:")
        print("  SPACE: 게임 시작")
        print("  R: 재시작") 
        print("  Q: 종료")
        print("-" * 50)
    
    def start_game(self):
        """게임 시작 - 카운트다운 시작"""
        if self.game_state == 'waiting' or self.game_state == 'result':
            print("🎯 게임 시작! 3초 후 녹화 시작...")
            self.reset_game_data()
            self.game_state = 'countdown'
            self.countdown_start_time = time.time()
    
    def reset_game_data(self):
        """게임 데이터 초기화"""
        self.angle_calculator.clear_data()
        self.final_score = None
        self.final_grade = None
    
    def update_game_state(self):
        """게임 상태 업데이트"""
        current_time = time.time()
        
        if self.game_state == 'countdown':
            # 카운트다운 체크
            if self.countdown_start_time is not None:
                elapsed = current_time - self.countdown_start_time 
                if elapsed >= self.countdown_duration:
                    self.game_state = 'recording'
                    self.recording_start_time = current_time
                    print("🔴 22초 댄스 녹화 시작!")
        
        elif self.game_state == 'recording':
            # 녹화 시간 체크
            if self.recording_start_time is not None:
                elapsed = current_time - self.recording_start_time
                if elapsed >= self.recording_duration:
                    self.game_state = 'processing'
                    print("⏹️ 녹화 완료! 점수 계산 중...")
                    self.process_dance_score()
    
    def process_dance_score(self):
        """댄스 점수 처리 (DTW + 점수 계산)"""
        try:
            # 1. 사용자 각도 데이터 가져오기
            user_angles = self.angle_calculator.get_angles_as_numpy()
            
            if user_angles is None or len(user_angles) == 0:
                print("❌ 녹화된 데이터가 없습니다.")
                self.game_state = 'waiting'
                return
            
            print(f"📊 사용자 데이터: {len(user_angles)}프레임")
            
            # 2. DTW 정렬
            print("🔄 DTW 시간 동기화 중...")
            aligned_ref, aligned_user = self.dtw_matcher.align_sequences(user_angles)
            
            if aligned_ref is None or aligned_user is None:
                print("❌ DTW 정렬에 실패했습니다.")
                self.game_state = 'waiting'
                return
            
            print(f"✅ DTW 정렬 완료: {len(aligned_ref)}프레임")
            
            # 3. 점수 계산
            print("🎯 점수 계산 중...")
            self.final_score = self.score_calculator.calculate_final_score(aligned_ref, aligned_user)
            
            if self.final_score is None:
                print("❌ 점수 계산에 실패했습니다.")
                self.game_state = 'waiting'
                return
            
            self.final_grade = self.score_calculator.get_score_grade(self.final_score)
            
            print(f"🎉 점수 계산 완료: {self.final_score:.1f}점 (등급: {self.final_grade})")
            self.game_state = 'result'
            
        except Exception as e:
            print(f"❌ 처리 중 오류 발생: {e}")
            self.game_state = 'waiting'
    
    def process_frame(self, frame):
        """프레임 처리 및 게임 상태에 따른 표시"""
        # 프레임 뒤집기 (거울 효과)
        frame = cv2.flip(frame, 1)
        
        # 각도 계산 (녹화 중일 때만 데이터 저장)
        processed_frame, angles = self.angle_calculator.extract_angles_from_frame(frame)
        
        if angles and self.game_state == 'recording':
            # 녹화 중일 때만 데이터 저장
            if self.recording_start_time is not None:
                timestamp = time.time() - self.recording_start_time
                self.angle_calculator.add_frame_data(angles, timestamp)
        
        # 게임 상태에 따른 UI 표시
        if self.game_state == 'waiting':
            processed_frame = self.draw_waiting_screen(processed_frame)
        
        elif self.game_state == 'countdown':
            processed_frame = self.draw_countdown(processed_frame)
        
        elif self.game_state == 'recording':
            processed_frame = self.draw_recording_screen(processed_frame)
        
        elif self.game_state == 'processing':
            processed_frame = self.draw_processing_screen(processed_frame)
        
        elif self.game_state == 'result':
            processed_frame = self.draw_result_screen(processed_frame)
        
        return processed_frame
    
    def draw_waiting_screen(self, frame):
        """대기 화면 그리기"""
        height, width = frame.shape[:2]
        
        # 반투명 오버레이
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # 제목
        title = "Candy Dance Challenge"
        self.draw_centered_text(frame, title, height//2 - 100, 2.0, (255, 255, 0), 3)
        
        # 안내 메시지
        messages = [
            "Press SPACE to start dancing!",
            "Follow the candy dance for 22 seconds",
            "Get the highest score possible!"
        ]
        
        for i, message in enumerate(messages):
            y_pos = height//2 - 20 + i * 50
            self.draw_centered_text(frame, message, y_pos, 1.0, (255, 255, 255), 2)
        
        return frame
    
    def draw_countdown(self, frame):
        """카운트다운 화면 그리기"""
        height, width = frame.shape[:2]
        
        # 반투명 오버레이
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # 카운트다운 숫자
        if self.countdown_start_time is not None:
            elapsed = time.time() - self.countdown_start_time
            remaining = max(0, self.countdown_duration - elapsed)
        
            if remaining > 0:
                countdown_text = str(int(remaining) + 1)
                self.draw_centered_text(frame, countdown_text, height//2, 8.0, (0, 255, 255), 5)
            else:
                self.draw_centered_text(frame, "START!", height//2, 4.0, (0, 255, 0), 5)
        else:
            self.draw_centered_text(frame, "3", height//2, 8.0, (0, 255, 255), 5)

        return frame
    
    def draw_recording_screen(self, frame):
        """녹화 화면 그리기"""
        height, width = frame.shape[:2]
        
        # 녹화 진행률 계산
        if self.recording_start_time is not None:
            elapsed = time.time() - self.recording_start_time
            remaining = max(0, self.recording_duration - elapsed)
            progress = (elapsed / self.recording_duration) * 100
        else:
            elapsed = 0
            remaining = self.recording_duration
            progress = 0
        
        # 상단 녹화 정보
        cv2.rectangle(frame, (20, 20), (width-20, 120), (0, 0, 0), -1)
        cv2.rectangle(frame, (20, 20), (width-20, 120), (0, 0, 255), 3)
        
        # 녹화 상태
        cv2.putText(frame, "🔴 RECORDING", (30, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        
        # 남은 시간
        cv2.putText(frame, f"Time Remaining: {remaining:.1f}s", (30, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 진행률 바
        bar_width = width - 60
        bar_x = 30
        bar_y = 90
        bar_height = 20
        
        # 배경 바
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        
        # 진행률 바
        progress_width = int(bar_width * progress / 100)
        if progress_width > 0:
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), (0, 0, 255), -1)
        
        # 데이터 수집 정보
        data_info = self.angle_calculator.get_data_info()
        cv2.putText(frame, f"Frames: {data_info['frame_count']}", 
                   (width-150, height-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def draw_processing_screen(self, frame):
        """처리 화면 그리기"""
        height, width = frame.shape[:2]
        
        # 반투명 오버레이
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # 처리 중 메시지
        self.draw_centered_text(frame, "Analyzing Your Dance...", height//2 - 50, 2.0, (0, 255, 255), 3)
        self.draw_centered_text(frame, "Please Wait", height//2 + 20, 1.5, (255, 255, 255), 2)
        
        # 로딩 애니메이션 (점 3개)
        dots = "." * (int(time.time() * 2) % 4)
        self.draw_centered_text(frame, dots, height//2 + 70, 2.0, (255, 255, 255), 2)
        
        return frame
    
    def draw_result_screen(self, frame):
        """결과 화면 그리기"""
        height, width = frame.shape[:2]
        
        # 반투명 오버레이
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # 결과 제목
        self.draw_centered_text(frame, "🎉 Dance Complete! 🎉", height//2 - 120, 2.0, (0, 255, 255), 3)
        
        # 점수 표시
        if self.final_score is not None:
            score_text = f"Final Score: {self.final_score:.1f}"
            score_color = self.get_score_color(self.final_score)
            self.draw_centered_text(frame, score_text, height//2 - 60, 2.5, score_color, 4)
            
            # 등급 표시
            grade_text = f"Grade: {self.final_grade}"
            self.draw_centered_text(frame, grade_text, height//2 - 10, 2.0, score_color, 3)
        
        # 재시작 안내
        restart_messages = [
            "Press 'R' to restart",
            "Press 'Q' to quit"
        ]
        
        for i, message in enumerate(restart_messages):
            y_pos = height//2 + 60 + i * 40
            self.draw_centered_text(frame, message, y_pos, 1.0, (200, 200, 200), 2)
        
        return frame
    
    def draw_centered_text(self, frame, text, y_pos, font_scale, color, thickness):
        """중앙 정렬 텍스트 그리기"""
        height, width = frame.shape[:2]
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        text_x = (width - text_size[0]) // 2
        cv2.putText(frame, text, (text_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    
    def get_score_color(self, score):
        """점수에 따른 색상 반환"""
        if score >= 90:
            return (0, 255, 0)      # 초록 (S등급)
        elif score >= 80:
            return (0, 255, 255)    # 노랑 (A등급)
        elif score >= 70:
            return (0, 165, 255)    # 주황 (B등급)
        elif score >= 60:
            return (0, 100, 255)    # 빨강-주황 (C등급)
        else:
            return (0, 0, 255)      # 빨강 (D등급)
    
    def handle_key_input(self, key):
        """키 입력 처리"""
        if key == ord(' '):  # 스페이스바: 게임 시작
            if self.game_state in ['waiting', 'result']:
                self.start_game()
        
        elif key == ord('r'):  # R키: 재시작
            self.game_state = 'waiting'
            self.reset_game_data()
            print("🔄 게임 재시작")
        
        elif key == ord('q'):  # Q키: 종료
            return False
        
        return True
    
    def run(self):
        """메인 게임 루프"""
        print("🚀 Hard Demo 시작!")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ 웹캠을 읽을 수 없습니다.")
                    break
                
                # 게임 상태 업데이트
                self.update_game_state()
                
                # 프레임 처리
                processed_frame = self.process_frame(frame)
                
                # 화면 표시
                cv2.imshow('Candy Dance Hard Demo', processed_frame)
                
                # 키 입력 처리
                key = cv2.waitKey(1) & 0xFF
                if not self.handle_key_input(key):
                    break
        
        except KeyboardInterrupt:
            print("\n사용자에 의해 중단되었습니다.")
        
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """리소스 정리"""
        print("\n🎵 Hard Demo 종료 중...")
        
        if hasattr(self, 'cap'):
            self.cap.release()
        
        cv2.destroyAllWindows()
        
        if hasattr(self, 'angle_calculator'):
            self.angle_calculator.cleanup()
        
        print("✅ 모든 리소스 정리 완료")
        
        # 최종 결과 출력
        if self.final_score is not None:
            print(f"\n🎊 최종 결과:")
            print(f"   점수: {self.final_score:.1f}점")
            print(f"   등급: {self.final_grade}")

# 메인 실행
if __name__ == "__main__":
    try:
        demo = HardDemoMain("candy_angles.csv")
        demo.run()
    except KeyboardInterrupt:
        print("\n프로그램이 중단되었습니다.")
    except Exception as e:
        print(f"오류 발생: {e}")