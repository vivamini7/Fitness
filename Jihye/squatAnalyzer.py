import cv2
import mediapipe as mp
import numpy as np
import math
import time

class SquatAnalyzer:
    """스쿼트 자세 분석 시스템"""
    
    def __init__(self, target_reps=10):
        # MediaPipe 설정
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        
        # 웹캠 설정
        self.cap = cv2.VideoCapture(0)
        
        # 설정값
        self.target_reps = target_reps
        self.stability_threshold = 2.0
        self.stability_duration = 1.0
        self.measurement_delay = 2.0
        
        # 상태 변수 초기화
        self.reset_all_variables()
    
    def reset_all_variables(self):
        """전체 변수 초기화"""
        self.counter = 0
        self.stage = None
        self.final_score = None
        self.rep_scores = []
        self.current_feedback = None
        self.reset_rep_variables()
    
    def reset_rep_variables(self):
        """회차별 변수 초기화"""
        self.angle_history = []
        self.is_stable = False
        self.countdown_active = False
        self.countdown_start_time = None
        self.measurement_triggered = False
        self.score_saved = False
    
    # ==================== 체크포인트 계산 함수들 ====================
    
    @staticmethod
    def calculate_angle(a, b, c):
        """세 점으로 각도 계산"""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
            
        return angle
    
    @staticmethod
    def calculate_trunk_angle(shoulder, hip):
        """상체 각도 계산"""
        dx = shoulder[0] - hip[0]
        dy = shoulder[1] - hip[1]
        angle = math.degrees(math.atan2(dx, dy))
        return abs(angle)
    
    def calculate_hip_angle(self, shoulder, hip, knee):
        """엉덩이 각도 계산"""
        return 180 - self.calculate_angle(shoulder, hip, knee)
    
    def calculate_knee_angle(self, hip, knee, ankle):
        """무릎 각도 계산"""
        return 180 - self.calculate_angle(hip, knee, ankle)
    
    @staticmethod
    def calculate_knee_over_toe_distance(knee, ankle):
        """무릎-발가락 x좌표의 차이"""
        return knee[0] - ankle[0]
    
    @staticmethod
    def calculate_knee_valgus_ratio(left_knee, right_knee, left_ankle, right_ankle):
        """무릎 안쪽 기울어짐 비율 계산"""
        knee_distance = math.sqrt((left_knee[0] - right_knee[0])**2 + 
                                (left_knee[1] - right_knee[1])**2)
        ankle_distance = math.sqrt((left_ankle[0] - right_ankle[0])**2 + 
                                 (left_ankle[1] - right_ankle[1])**2)
        
        if ankle_distance == 0:
            return 0.5
        
        return knee_distance / ankle_distance
    
    # ==================== 점수 계산 함수들 ====================
    
    @staticmethod
    def score_trunk_angle(trunk_angle):
        """상체 각도 점수 (0-30도가 이상적)"""
        if trunk_angle <= 15:
            return 100
        elif trunk_angle <= 30:
            return 100 - (trunk_angle - 15) * 4
        elif trunk_angle <= 45:
            return 40 - (trunk_angle - 30) * 2
        else:
            return 0
    
    @staticmethod
    def score_hip_angle(hip_angle):
        """엉덩이 각도 점수 (90-110도가 이상적)"""
        if 90 <= hip_angle <= 110:
            return 100
        elif 80 <= hip_angle < 90 or 110 < hip_angle <= 120:
            return 100 - abs(hip_angle - 100) * 2
        elif 70 <= hip_angle < 80 or 120 < hip_angle <= 130:
            return 60 - abs(hip_angle - 100) * 1.5
        else:
            return 0
    
    @staticmethod
    def score_knee_angle(knee_angle):
        """무릎 각도 점수 (85-95도가 이상적)"""
        if 85 <= knee_angle <= 95:
            return 100
        elif 75 <= knee_angle < 85 or 95 < knee_angle <= 105:
            return 100 - abs(knee_angle - 90) * 4
        elif 65 <= knee_angle < 75 or 105 < knee_angle <= 115:
            return 60 - abs(knee_angle - 90) * 2
        else:
            return 0
    
    @staticmethod
    def score_knee_over_toe(knee_over_toe_distance):
        """무릎-발가락 전방이동 점수 (음수가 이상적)"""
        if knee_over_toe_distance <= 0:
            return 100
        elif knee_over_toe_distance <= 0.05:
            return 100 - knee_over_toe_distance * 800
        elif knee_over_toe_distance <= 0.1:
            return 60 - (knee_over_toe_distance - 0.05) * 1200
        else:
            return 0
    
    @staticmethod
    def score_knee_valgus(knee_valgus_ratio):
        """무릎 벌어짐 점수 (1.0 이상이 이상적)"""
        if knee_valgus_ratio >= 1.0:
            return 100
        elif knee_valgus_ratio >= 0.9:
            return 100 - (1.0 - knee_valgus_ratio) * 500
        elif knee_valgus_ratio >= 0.8:
            return 50 - (0.9 - knee_valgus_ratio) * 300
        else:
            return 0
    
    @staticmethod
    def get_feedback_text(total_score):
        """점수에 따른 피드백 텍스트"""
        if total_score >= 90:
            return "Perfect"
        elif total_score >= 60:
            return "Good"
        else:
            return "Bad"
    
    @staticmethod
    def get_feedback_color(feedback):
        """피드백에 따른 색상 반환 (BGR 형식)"""
        color_map = {
            "Perfect": (0, 255, 0),    # 초록색
            "Good": (0, 255, 255),     # 노란색
            "Bad": (0, 0, 255)         # 빨간색
        }
        return color_map.get(feedback, (255, 255, 255))  # 기본값: 흰색
    
    # ==================== 안정화 감지 ====================
    
    def check_angle_stability(self):
        """무릎 각도 변화율이 임계값 이하로 지정 시간동안 유지되는지 확인"""
        if len(self.angle_history) < 2:
            return False
        
        current_time = self.angle_history[-1][0]
        
        # 지정 시간 내의 각도 기록만 필터링
        recent_angles = [
            (t, angle) for t, angle in self.angle_history 
            if current_time - t <= self.stability_duration
        ]
        
        if len(recent_angles) < 2:
            return False
        
        # 각도 변화율 계산
        max_change_rate = 0
        for i in range(1, len(recent_angles)):
            time_diff = recent_angles[i][0] - recent_angles[i-1][0]
            if time_diff > 0:
                angle_diff = abs(recent_angles[i][1] - recent_angles[i-1][1])
                change_rate = angle_diff / time_diff
                max_change_rate = max(max_change_rate, change_rate)
        
        return max_change_rate <= self.stability_threshold
    
    # ==================== 자세 분석 ====================
    
    def extract_landmarks(self, landmarks):
        """랜드마크 추출"""
        return {
            'shoulder': [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y],
            'hip': [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,
                   landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y],
            'left_knee': [landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                         landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y],
            'left_ankle': [landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                          landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y],
            'right_knee': [landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                          landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].y],
            'right_ankle': [landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                           landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
        }
    
    def analyze_pose(self, landmarks):
        """스쿼트 자세 분석 및 점수 계산"""
        points = self.extract_landmarks(landmarks)
        
        # 체크포인트 값 계산
        trunk_angle = self.calculate_trunk_angle(points['shoulder'], points['hip'])
        hip_angle = self.calculate_hip_angle(points['shoulder'], points['hip'], points['left_knee'])
        knee_angle = self.calculate_knee_angle(points['hip'], points['left_knee'], points['left_ankle'])
        knee_over_toe_distance = self.calculate_knee_over_toe_distance(points['left_knee'], points['left_ankle'])
        knee_valgus_ratio = self.calculate_knee_valgus_ratio(
            points['left_knee'], points['right_knee'], 
            points['left_ankle'], points['right_ankle']
        )
        
        angles_info = {
            'trunk': trunk_angle,
            'hip': hip_angle,
            'knee': knee_angle,
            'knee_over_toe': knee_over_toe_distance,
            'knee_valgus': knee_valgus_ratio
        }
        
        # 점수 계산 (측정 완료 시에만)
        if self.stage == "DOWN" and self.measurement_triggered:
            scores = {
                'trunk': self.score_trunk_angle(trunk_angle),
                'hip': self.score_hip_angle(hip_angle),
                'knee': self.score_knee_angle(knee_angle),
                'knee_over_toe': self.score_knee_over_toe(knee_over_toe_distance),
                'knee_valgus': self.score_knee_valgus(knee_valgus_ratio)
            }
            
            total_score = sum(scores.values()) / len(scores)
            feedback = self.get_feedback_text(total_score)
            
            return {
                'scores': scores,
                'total_score': total_score,
                'feedback': feedback,
                'angles': angles_info
            }
        
        return {'angles': angles_info}
    
    # ==================== 상태 관리 ====================
    
    def update_stage(self, knee_angle):
        """상태 업데이트 및 회차 관리"""
        prev_stage = self.stage
        
        # 상태 판정
        if knee_angle > 169:
            self.stage = "UP"
            # DOWN에서 UP으로 전환시 회차 완료
            if prev_stage == "DOWN" and self.score_saved:
                print(f"스쿼트 {self.counter}회 완료!")
                self.check_set_completion()
            self.reset_rep_variables()
        
        elif knee_angle <= 90 and self.stage == 'UP':
            self.stage = "DOWN"
            self.counter += 1
            print(f"스쿼트 {self.counter}회 시작!")
            self.reset_rep_variables()
    
    def process_down_stage(self, current_time, knee_angle):
        """DOWN 상태에서의 안정화 감지 및 측정 처리"""
        # 각도 기록 추가
        self.angle_history.append((current_time, knee_angle))
        
        # 오래된 기록 제거 (5초 이상)
        self.angle_history = [
            (t, a) for t, a in self.angle_history 
            if current_time - t <= 5.0
        ]
        
        # 안정화 체크
        if not self.is_stable and not self.countdown_active:
            if self.check_angle_stability():
                self.is_stable = True
                self.countdown_active = True
                self.countdown_start_time = current_time
                print("자세가 안정화됨! 측정 준비 중...")
        
        # 카운트다운 및 측정 처리
        if self.countdown_active and not self.measurement_triggered:
            elapsed = current_time - self.countdown_start_time
            if elapsed >= self.measurement_delay:
                self.measurement_triggered = True
                print("측정 완료!")
    
    def check_set_completion(self):
        """1세트 완료 확인"""
        if self.counter == self.target_reps and len(self.rep_scores) == self.target_reps:
            self.final_score = sum(self.rep_scores) / len(self.rep_scores)
            print(f"\n=== 1세트 완료! ===")
            print(f"최종 점수: {self.final_score:.1f}")
            print("'r' 키를 눌러 재시작하거나 'q' 키로 종료하세요.")
            return True
        return False
    
    # ==================== 화면 표시 ====================
    
    def draw_info_box(self, image):
        """정보 표시 박스 그리기"""
        cv2.rectangle(image, (0, 0), (400, 200), (245, 117, 16), -1)
        
        # 현재 횟수 표시
        cv2.putText(image, 'REPS', (15, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, f"{self.counter}/{self.target_reps}", 
                    (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
        
        # 현재 상태 표시
        cv2.putText(image, 'STAGE', (150, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        if self.stage:
            cv2.putText(image, self.stage, (150, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
    
    def draw_status_feedback(self, image):
        """상태별 피드백 표시"""
        if self.countdown_active and not self.measurement_triggered:
            self._draw_countdown(image)
        elif self.current_feedback and self.measurement_triggered:
            self._draw_feedback(image)
        elif self.stage == "DOWN" and not self.is_stable:
            self._draw_stabilizing(image)
    
    def _draw_countdown(self, image):
        """카운트다운 표시"""
        elapsed = time.time() - self.countdown_start_time
        remaining = 3.0 - elapsed
        countdown_text = str(int(remaining) + 1) if remaining > 0 else "측정중..."
        
        cv2.putText(image, 'MEASURING', (15, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, countdown_text, (15, 135), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3, cv2.LINE_AA)
    
    def _draw_feedback(self, image):
        """피드백 표시"""
        cv2.putText(image, 'FEEDBACK', (15, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        
        feedback_color = self.get_feedback_color(self.current_feedback)
        cv2.putText(image, self.current_feedback, (15, 135), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, feedback_color, 3, cv2.LINE_AA)
    
    def _draw_stabilizing(self, image):
        """안정화 대기 표시"""
        cv2.putText(image, 'STABILIZING', (15, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, 'Hold pose...', (15, 135), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
    
    def draw_final_score(self, image):
        """최종 점수 표시"""
        if self.final_score is not None:
            cv2.rectangle(image, (0, 220), (400, 320), (0, 0, 255), -1)
            cv2.putText(image, 'FINAL SCORE', (15, 250), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f"{self.final_score:.1f}/100", (15, 290), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
    
    # ==================== 메인 실행 ====================
    
    def run(self):
        """메인 실행 루프"""
        with self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                
                if not ret:
                    print("웹캠을 읽을 수 없습니다.")
                    break
                
                # 이미지 전처리
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                
                # 포즈 감지
                results = pose.process(image)
                
                # 이미지 후처리
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                # 랜드마크 분석
                if results.pose_landmarks:
                    self._process_landmarks(results.pose_landmarks.landmark)
                
                # 화면 표시
                self._draw_interface(image, results)
                
                # 키 입력 처리
                if self._handle_key_input():
                    break
        
        self._cleanup()
    
    def _process_landmarks(self, landmarks):
        """랜드마크 처리"""
        try:
            # 기본 무릎 각도 계산 (상태 판정용)
            hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,
                   landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]
            left_knee = [landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            left_ankle = [landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                         landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            
            current_time = time.time()
            angle_knee_raw = self.calculate_angle(hip, left_knee, left_ankle)
            knee_angle = 180 - angle_knee_raw
            
            # 상태 업데이트
            self.update_stage(knee_angle)
            
            # DOWN 상태 처리
            if self.stage == "DOWN":
                self.process_down_stage(current_time, knee_angle)
            
            # 자세 분석
            analysis = self.analyze_pose(landmarks)
            
            # 점수 저장
            if (analysis and 'total_score' in analysis and 
                self.measurement_triggered and not self.score_saved):
                
                rep_score = analysis['total_score']
                self.rep_scores.append(rep_score)
                self.current_feedback = analysis['feedback']
                print(f"이번 회차 점수: {rep_score:.1f} - {analysis['feedback']}")
                self.score_saved = True
                
        except Exception as e:
            print(f"분석 오류: {e}")
    
    def _draw_interface(self, image, results):
        """화면 인터페이스 그리기"""
        # 정보 박스
        self.draw_info_box(image)
        
        # 상태별 피드백
        self.draw_status_feedback(image)
        
        # 최종 점수
        self.draw_final_score(image)
        
        # 포즈 랜드마크
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )
        
        # 화면 출력
        cv2.imshow('스쿼트 자세 분석', image)
    
    def _handle_key_input(self):
        """키 입력 처리"""
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            return True
        elif key == ord('r') and self.final_score is not None:
            self.reset_all_variables()
            print("\n시스템 재시작!")
        return False
    
    def _cleanup(self):
        """정리 작업"""
        self.cap.release()
        cv2.destroyAllWindows()
        print("프로그램이 종료되었습니다.")

# ==================== 실행 ====================

if __name__ == "__main__":
    analyzer = SquatAnalyzer(target_reps=10)
    analyzer.run()