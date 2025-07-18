import cv2
import mediapipe as mp
import numpy as np
import math
import time

# MediaPipe 설정
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# 웹캠 설정
cap = cv2.VideoCapture(0)

# 스쿼트 카운터 변수들
counter = 0
stage = None
final_score = None

# 안정화 감지용 변수들
angle_history = []           # 최근 각도 기록 (시간, 각도) 튜플 리스트
stability_threshold = 2.0    # 각도 변화 임계값 (도/초)
stability_duration = 1.0     # 안정화 판정 시간 (초)
is_stable = False           # 안정화 상태 플래그

# 카운트다운 및 측정용 변수들
countdown_active = False    # 카운트다운 진행 중 플래그
countdown_start_time = None # 카운트다운 시작 시간
measurement_delay = 2.0     # 실제 측정까지 시간 (초)
measurement_triggered = False # 측정 완료 플래그
score_saved = False         # 점수 저장 완료 플래그

# 점수 추적 변수들
rep_scores = []          # 각 회차별 점수 저장
current_feedback = None  # 현재 회차의 피드백 (화면 표시용, 저장하지 않음)

# 사용자 설정
target_reps = 10         # 목표 횟수 (사용자가 설정)

# ==================== 체크포인트 계산 함수들 ====================

#Base: 각도 계산 함수
def calculate_angle(a, b, c):
    """세 점으로 각도 계산"""
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
        
    return angle

#Point1: Trunk Angle (상체 각도)
def calculate_trunk_angle(shoulder, hip):
    """어깨-엉덩이 선분의 기울기기"""
    dx = shoulder[0] - hip[0]
    dy = shoulder[1] - hip[1]
    angle = math.degrees(math.atan2(dx, dy))
    return abs(angle)

#Point2: Hip Angle (엉덩이 각도)
def calculate_hip_angle(shoulder, hip, knee):
    hip_angle = 180 - calculate_angle(shoulder, hip, knee)
    return hip_angle

#Point3: Knee Angle (무릎 각도)
def calculate_knee_angle(hip, knee, ankle):
    knee_angle = 180 - calculate_angle(hip, knee, ankle)
    return knee_angle

#Point4: Anterior Knee-over-toe Translation (무릎-발가락 전방 이동)
def calculate_knee_over_toe_distance(knee, ankle):
    """무릎-발가락 x좌표의 차이"""
    return knee[0] - ankle[0]

#Point5: Knee Valgus
def calculate_knee_valgus_ratio(left_knee, right_knee, left_ankle, right_ankle):
    """무릎 안쪽 기울어짐 비율 계산"""
    knee_distance = math.sqrt((left_knee[0] - right_knee[0])**2 + 
                        (left_knee[1] - right_knee[1])**2)
    ankle_distance = math.sqrt((left_ankle[0] - right_ankle[0])**2 + 
                        (left_ankle[1] - right_ankle[1])**2)

    if ankle_distance == 0:  # 발목 거리가 0인 경우 예외 처리
            return 0.5
    
    # 무릎 거리 / 발목 거리 비율 계산
    ratio = knee_distance / ankle_distance
    return ratio

# ==================== 점수 측정 타이밍 계산 함수 ======================
def check_angle_stability(angle_history, threshold, duration):
    '''무릎 각도 변화율이 임계값 이하로 지정 시간동안 유지되는지 확인'''

    if len(angle_history) < 2:
        return False
    
    current_time = angle_history[-1][0]
    
    # duration 시간 내의 각도 기록만 필터링
    recent_angles = [
        (t, angle) for t, angle in angle_history 
        if current_time - t <= duration
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
    
    return max_change_rate <= threshold


# ==================== 점수 계산 함수들 (임시 기준) ====================

def score_trunk_angle(trunk_angle):
    """상체 각도 점수 (0-30도가 이상적)"""
    if trunk_angle <= 15:
        return 100
    elif trunk_angle <= 30:
        return 100 - (trunk_angle - 15) * 4  # 15도 초과시 점수 감소
    elif trunk_angle <= 45:
        return 40 - (trunk_angle - 30) * 2   # 30도 초과시 더 큰 감소
    else:
        return 0

def score_hip_angle(hip_angle):
    """엉덩이 각도 점수 (80-120도가 이상적)"""
    if 90 <= hip_angle <= 110:
        return 100
    elif 80 <= hip_angle < 90 or 110 < hip_angle <= 120:
        return 100 - abs(hip_angle - 100) * 2  # 최적 범위에서 벗어날수록 감소
    elif 70 <= hip_angle < 80 or 120 < hip_angle <= 130:
        return 60 - abs(hip_angle - 100) * 1.5
    else:
        return 0

def score_knee_angle(knee_angle):
    """무릎 각도 점수 (80-100도가 이상적)"""
    if 85 <= knee_angle <= 95:
        return 100
    elif 75 <= knee_angle < 85 or 95 < knee_angle <= 105:
        return 100 - abs(knee_angle - 90) * 4  # 90도에서 벗어날수록 감소
    elif 65 <= knee_angle < 75 or 105 < knee_angle <= 115:
        return 60 - abs(knee_angle - 90) * 2
    else:
        return 0

def score_knee_over_toe(knee_over_toe_distance):
    """무릎-발가락 전방이동 점수 (음수가 이상적)"""
    if knee_over_toe_distance <= 0:
        return 100  # 무릎이 발목보다 뒤에 있으면 완벽
    elif knee_over_toe_distance <= 0.05:  # 약간 앞으로 나간 경우
        return 100 - knee_over_toe_distance * 800  # 0.05까지는 점진적 감소
    elif knee_over_toe_distance <= 0.1:
        return 60 - (knee_over_toe_distance - 0.05) * 1200  # 더 큰 감소
    else:
        return 0

def score_knee_valgus(knee_valgus_ratio):
    """무릎 벌어짐 점수 (무릎 거리가 발목 거리보다 좁지 않아야 함)"""
    if knee_valgus_ratio >= 1.0:  # 무릎이 발목보다 넓거나 같으면 완벽
        return 100
    elif knee_valgus_ratio >= 0.9:  # 약간 좁은 경우 (90% 이상)
        return 100 - (1.0 - knee_valgus_ratio) * 500  # 90-100% 범위에서 점진적 감소
    elif knee_valgus_ratio >= 0.8:  # 더 좁은 경우 (80-90%)
        return 50 - (0.9 - knee_valgus_ratio) * 300   # 더 큰 점수 감소
    else:  # 매우 좁은 경우 (80% 미만)
        return 0
    

# ==================== 피드백 함수 ====================

def get_feedback_text(total_score):
    """전체 점수에 따른 피드백 텍스트"""
    if total_score >= 90:
        return "Perfect"
    elif total_score >= 60:
        return "Good"
    else:
        return "Bad"

# ==================== 메인 분석 함수 ====================

def analyze_squat_pose(landmarks, stage, measurement_triggered):
    """스쿼트 자세 분석 및 점수 계산"""
    
    # 랜드마크 추출
    '''양쪽 모두 있는 것은 왼쪽을 기본으로 함'''

    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
           landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
             landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                  landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
    right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
    
    # 체크포인트 값 계산
    trunk_angle = calculate_trunk_angle(shoulder, hip)
    hip_angle = calculate_hip_angle(shoulder, hip, left_knee)
    knee_angle = calculate_knee_angle(hip, left_knee, left_ankle)
    knee_over_toe_distance = calculate_knee_over_toe_distance(left_knee, left_ankle)
    knee_valgus_ratio = calculate_knee_valgus_ratio(left_knee, right_knee, left_ankle, right_ankle)

    #점수 계산
    if stage == "DOWN" and measurement_triggered:
        scores = {
            'trunk': score_trunk_angle(trunk_angle),
            'hip': score_hip_angle(hip_angle),
            'knee': score_knee_angle(knee_angle),
            'knee_over_toe': score_knee_over_toe(knee_over_toe_distance),
            'knee_valgus': score_knee_valgus(knee_valgus_ratio)
        }
        
        total_score = sum(scores.values()) / len(scores)
        feedback = get_feedback_text(total_score)
        
        return {
            'scores': scores,
            'total_score': total_score,
            'feedback': feedback,
            'angles': {
                'trunk': trunk_angle,
                'hip': hip_angle,
                'knee': knee_angle,
                'knee_over_toe': knee_over_toe_distance,
                'knee_valgus' : knee_valgus_ratio
            }
        }
    
    # 점수 계산하지 않을 때는 각도 정보만 반환
    return {
        'angles': {
            'trunk': trunk_angle,
            'hip': hip_angle,
            'knee': knee_angle,
            'knee_over_toe': knee_over_toe_distance,
            'knee_valgus' : knee_valgus_ratio
        }
    }

# ============================== 초기화 함수 ==============================
def reset_rep_variables():
    """회차별 변수 초기화"""
    global angle_history, is_stable, countdown_active, measurement_triggered, score_saved
    angle_history = []
    is_stable = False
    countdown_active = False
    measurement_triggered = False
    score_saved = False

def reset_all_variables():
    """전체 변수 초기화 (재시작용)"""
    global counter, stage, rep_scores, current_feedback, final_score
    counter = 0
    stage = None
    rep_scores = []
    current_feedback = None
    final_score = None
    reset_rep_variables()

def check_set_completion():
    """1세트 완료 확인 및 처리"""
    global final_score
    if counter == target_reps and len(rep_scores) == target_reps:
        final_score = sum(rep_scores) / len(rep_scores)
        print(f"\n=== 1세트 완료! ===")
        print(f"최종 점수: {final_score:.1f}")
        print("'r' 키를 눌러 재시작하거나 'q' 키로 종료하세요.")
        return True
    return False

# ==================== 메인 웹캠 루프 ====================

# MediaPipe 포즈 감지 시작
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            print("웹캠을 읽을 수 없습니다.")
            break
        
        # BGR을 RGB로 변환
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # 포즈 감지
        results = pose.process(image)
    
        # RGB를 BGR로 다시 변환
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # 랜드마크 추출 및 분석
        try:
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # 기본 각도 계산 (상태 판정용)
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                
                current_time = time.time()
                angle_knee_raw = calculate_angle(hip, left_knee, left_ankle)
                knee_angle = 180 - angle_knee_raw
                                
                # 이전 상태 저장
                prev_stage = stage
                
                # 상태 판정 (참고 코드와 동일한 로직)
                if knee_angle > 169:
                    stage = "UP"
                    if prev_stage == "DOWN" and score_saved:
                        print(f"스쿼트 {counter}회 완료!")
                        check_set_completion()
                    reset_rep_variables()

                if knee_angle <= 90 and stage == 'UP':
                    stage = "DOWN"
                    counter += 1
                    print(f"스쿼트 {counter}회 시작!")
                    reset_rep_variables()
                
                # DOWN 상태에서 안정화 감지 및 측정 처리
                if stage == "DOWN":
                    angle_history.append((current_time, knee_angle))
                    
                    # 오래된 기록 제거 (5초 이상)
                    angle_history = [(t, a) for t, a in angle_history if current_time - t <= 5.0]
                    
                    # 안정화 체크
                    if not is_stable and not countdown_active:
                        if check_angle_stability(angle_history, stability_threshold, stability_duration):
                            is_stable = True
                            countdown_active = True
                            countdown_start_time = current_time
                            print("자세가 안정화됨! 측정 준비 중...")
                    
                    # 카운트다운 및 측정 처리
                    if countdown_active and not measurement_triggered:
                        elapsed = current_time - countdown_start_time
                        
                        # 2초 후 실제 측정
                        if elapsed >= measurement_delay:
                            measurement_triggered = True
                            print(f"측정 완료!")

                # 자세 분석
                analysis = analyze_squat_pose(landmarks, stage, measurement_triggered)
                
                # 점수 저장
                if analysis and 'total_score' in analysis and measurement_triggered and not score_saved:
                    # 5개 체크포인트 평균 점수를 회차별 점수로 저장
                    rep_score = analysis['total_score']
                    rep_scores.append(rep_score)
                    current_feedback = analysis['feedback']
                    print(f"이번 회차 점수: {rep_score:.1f} - {analysis['feedback']}")
                    score_saved = True                
                        
        except Exception as e:
            print(f"분석 오류: {e}")
            pass
        
        # ==================== 화면 표시 ====================
        
        # 상태 박스 생성
        cv2.rectangle(image, (0, 0), (400, 200), (245, 117, 16), -1)
        
        # 현재 횟수 표시
        cv2.putText(image, 'REPS', (15, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, f"{counter}/{target_reps}", 
                    (15, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
        
        # 현재 상태 표시
        cv2.putText(image, 'STAGE', (150, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        if stage:
            cv2.putText(image, stage, 
                        (150, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
        
        # 현재 피드백 표시
        if countdown_active and not measurement_triggered:
            # 카운트다운 표시
            elapsed = time.time() - countdown_start_time
            remaining = 3.0 - elapsed
            if remaining > 0:
                countdown_text = str(int(remaining) + 1)  # 3, 2, 1 표시
            else:
                countdown_text = "측정중..."
                        
            cv2.putText(image, 'MEASURING', (15, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(image, countdown_text, 
                        (15, 135), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3, cv2.LINE_AA)
            
        elif current_feedback and measurement_triggered:
            # 측정 완료 후 피드백 표시
            cv2.putText(image, 'FEEDBACK', (15, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            # 피드백에 따른 색상 설정
            if current_feedback == "Perfect":
                feedback_color = (0, 255, 0)  # 초록색
            elif current_feedback == "Good":
                feedback_color = (0, 255, 255)  # 노란색
            else:  # "Bad"
                feedback_color = (0, 0, 255)  # 빨간색

            cv2.putText(image, current_feedback, 
                        (15, 135), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, feedback_color, 3, cv2.LINE_AA)

        elif stage == "DOWN" and not is_stable:
            # 안정화 대기 중 표시
            cv2.putText(image, 'STABILIZING', (15, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(image, 'Hold pose...', 
                        (15, 135), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
        
        # 1세트 완료 시 최종 점수 표시
        if final_score is not None:
            cv2.rectangle(image, (0, 220), (400, 320), (0, 0, 255), -1)
            cv2.putText(image, 'FINAL SCORE', (15, 250), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f"{final_score:.1f}/100", 
                        (15, 290), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
        
        # 포즈 랜드마크 그리기
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2), 
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )
        
        # 화면 출력
        cv2.imshow('스쿼트 자세 분석', image)
        
        # 키 입력 처리
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r') and final_score is not None:
            reset_all_variables()
            print("\n시스템 재시작!")

# 정리
cap.release()
cv2.destroyAllWindows()
print("프로그램이 종료되었습니다.")