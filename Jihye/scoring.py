import cv2
import mediapipe as mp
import numpy as np
import math

# MediaPipe 설정
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# 각도 계산 함수
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

# 웹캠 설정
cap = cv2.VideoCapture(0)

# 스쿼트 카운터 변수들
counter = 0
stage = None

# 각도 추적 배열 (각 회차별 각도 저장)
angle_min = []           # 각 회차 동안의 무릎 각도들
angle_min_hip = []       # 각 회차 동안의 엉덩이 각도들
trunk_angles = []        # 각 회차 동안의 상체 각도들
knee_over_toe_values = []  # 각 회차 동안의 무릎-발가락 거리들
knee_distances = []      # 각 회차 동안의 무릎 간 거리들
ankle_distances = []     # 각 회차 동안의 발목 간 거리들

# 점수 추적 변수들
rep_scores = []          # 각 회차별 점수 저장
current_rep_scores = []  # 현재 회차의 점수들 (DOWN 상태 동안)
current_feedback = None  # 현재 회차의 피드백 (화면 표시용, 저장하지 않음)

# 최소/최대 각도 저장 (각 회차별)
min_ang = 0
max_ang = 0
min_ang_hip = 0
max_ang_hip = 0

# 사용자 설정
target_reps = 10         # 목표 횟수 (사용자가 설정)

# ==================== 체크포인트 계산 함수들 ====================

def calculate_trunk_angle(shoulder, hip):
    """상체 기울기 각도 계산"""
    dx = shoulder[0] - hip[0]
    dy = shoulder[1] - hip[1]
    angle = math.degrees(math.atan2(dx, dy))
    return abs(angle)

def calculate_knee_over_toe_distance(knee, ankle):
    """무릎-발가락 전방 이동 거리 계산"""
    return knee[0] - ankle[0]

def calculate_knee_valgus_distance(left_knee, right_knee):
    """양쪽 무릎 사이 거리 계산"""
    distance = math.sqrt((left_knee[0] - right_knee[0])**2 + 
                        (left_knee[1] - right_knee[1])**2)
    return distance

def calculate_ankle_distance(left_ankle, right_ankle):
    """양쪽 발목 사이 거리 계산"""
    distance = math.sqrt((left_ankle[0] - right_ankle[0])**2 + 
                        (left_ankle[1] - right_ankle[1])**2)
    return distance

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

def score_knee_valgus(knee_distance, ankle_distance):
    """무릎 벌어짐 점수 (무릎 거리가 발목 거리보다 좁지 않아야 함)"""
    if ankle_distance == 0:  # 발목 거리가 0인 경우 예외 처리
        return 50
    
    # 무릎 거리 / 발목 거리 비율 계산
    ratio = knee_distance / ankle_distance
    
    if ratio >= 1.0:  # 무릎이 발목보다 넓거나 같으면 완벽
        return 100
    elif ratio >= 0.9:  # 약간 좁은 경우 (90% 이상)
        return 100 - (1.0 - ratio) * 500  # 90-100% 범위에서 점진적 감소
    elif ratio >= 0.8:  # 더 좁은 경우 (80-90%)
        return 50 - (0.9 - ratio) * 300   # 더 큰 점수 감소
    else:  # 매우 좁은 경우 (80% 미만)
        return 0
    

# ==================== 피드백 함수 ====================

def get_feedback_text(total_score):
    """전체 점수에 따른 피드백 텍스트 (각 회차 DOWN시마다 생성)"""
    if total_score >= 90:
        return "Perfect"
    elif total_score >= 60:
        return "Good"
    else:
        return "Bad"

# ==================== 메인 분석 함수 ====================

def analyze_squat_pose(landmarks, stage):
    """스쿼트 자세 분석 및 점수 계산"""
    
    # 랜드마크 추출 (왼쪽 기준)
    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
           landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
             landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
    
    # 오른쪽 관절도 추출 (Knee Valgus 계산용)
    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                  landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
    right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
    
    # 기본 각도 계산
    angle_knee_raw = calculate_angle(hip, knee, ankle)
    angle_hip_raw = calculate_angle(shoulder, hip, knee)
    
    knee_angle = 180 - angle_knee_raw
    hip_angle = 180 - angle_hip_raw
    
    # 체크포인트 값 계산
    trunk_angle = calculate_trunk_angle(shoulder, hip)
    knee_over_toe_distance = calculate_knee_over_toe_distance(knee, ankle)
    knee_distance = calculate_knee_valgus_distance(knee, right_knee)
    ankle_distance = calculate_ankle_distance(ankle, right_ankle)
    
    # DOWN 상태일 때만 점수 계산
    if stage == "DOWN":
        scores = {
            'trunk': score_trunk_angle(trunk_angle),
            'hip': score_hip_angle(hip_angle),
            'knee': score_knee_angle(knee_angle),
            'knee_over_toe': score_knee_over_toe(knee_over_toe_distance),
            'knee_valgus': score_knee_valgus(knee_distance, ankle_distance)
        }
        
        total_score = sum(scores.values()) / len(scores)
        feedback = get_feedback_text(total_score)  # 피드백 생성 (화면 표시용)
        
        return {
            'scores': scores,
            'total_score': total_score,
            'feedback': feedback,
            'angles': {
                'trunk': trunk_angle,
                'hip': hip_angle,
                'knee': knee_angle,
                'knee_over_toe': knee_over_toe_distance,
                'knee_distance': knee_distance,
                'ankle_distance': ankle_distance
            }
        }
    
    # UP 상태일 때는 각도 정보만 반환
    return {
        'angles': {
            'trunk': trunk_angle,
            'hip': hip_angle,
            'knee': knee_angle,
            'knee_over_toe': knee_over_toe_distance,
            'knee_distance': knee_distance,
            'ankle_distance': ankle_distance
        }
    }
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
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                
                angle_knee_raw = calculate_angle(hip, knee, ankle)
                knee_angle = 180 - angle_knee_raw
                
                # 각도 추적 배열에 저장
                angle_min.append(angle_knee_raw)
                
                # 이전 상태 저장
                prev_stage = stage
                
                # 상태 판정 (참고 코드와 동일한 로직)
                if knee_angle > 169:
                    stage = "UP"
                if knee_angle <= 90 and stage == 'UP':
                    stage = "DOWN"
                    counter += 1
                    
                    # 1회 완료 처리
                    print(f"스쿼트 {counter}회 완료!")
                    
                    # 해당 회차의 최소 각도 계산 (참고 코드 방식)
                    min_ang = min(angle_min)
                    max_ang = max(angle_min)
                    print(f"무릎 각도 범위: {min_ang:.1f}° ~ {max_ang:.1f}°")
                    
                    # 현재 회차의 평균 점수 계산 및 저장
                    if current_rep_scores:
                        rep_score = sum(current_rep_scores) / len(current_rep_scores)
                        rep_scores.append(rep_score)
                        print(f"이번 회차 점수: {rep_score:.1f}")
                        current_rep_scores = []  # 다음 회차를 위해 초기화
                    
                    # 각도 배열 초기화 (다음 회차 준비)
                    angle_min = []
                    
                    # 1세트 완료 확인
                    if counter == target_reps:
                        final_score = sum(rep_scores) / len(rep_scores)
                        print(f"\n=== 1세트 완료! ===")
                        print(f"최종 점수: {final_score:.1f}")
                        print("'r' 키를 눌러 재시작하거나 'q' 키로 종료하세요.")
                
                # 자세 분석 (DOWN 상태일 때만)
                analysis = analyze_squat_pose(landmarks, stage)
                
                if analysis and 'feedback' in analysis:
                    # DOWN 상태: 피드백 저장 및 점수 수집
                    current_feedback = analysis['feedback']
                    current_rep_scores.append(analysis['total_score'])
                        
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
        
        # 현재 피드백 표시 (DOWN 상태일 때만)
        if current_feedback and stage == "DOWN":
            cv2.putText(image, 'FEEDBACK', (15, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(image, current_feedback, 
                        (15, 135), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
        
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
            # 재시작
            counter = 0
            stage = None
            rep_scores = []
            current_rep_scores = []
            current_feedback = None
            final_score = None
            angle_min = []
            min_ang = 0
            max_ang = 0
            print("\n시스템 재시작!")

# 정리
cap.release()
cv2.destroyAllWindows()
print("프로그램이 종료되었습니다.")