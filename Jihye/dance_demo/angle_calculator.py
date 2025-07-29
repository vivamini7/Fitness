import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time

class AngleCalculator:
    """메인 시스템에서 사용할 각도 계산 전용 클래스"""
    
    def __init__(self):
        # MediaPipe 초기화
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Pose 모델 설정
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
        
        # 관절 인덱스 (기준 코드와 동일)
        self.joint_indices = {
            "left_shoulder": 11,
            "right_shoulder": 12,
            "left_elbow": 13,
            "right_elbow": 14,
            "left_wrist": 15,
            "right_wrist": 16,
            "left_hip": 23,
            "right_hip": 24,
            "left_knee": 25,
            "right_knee": 26,
            "left_ankle": 27,
            "right_ankle": 28
        }
        
        # 녹화된 각도 데이터 저장
        self.recorded_data = []
    
    def calculate_angle(self, a, b):
        """
        두 벡터 사이의 각도 계산 (기준 코드와 완전히 동일한 방식)
        Args:
            a: 첫 번째 벡터
            b: 두 번째 벡터
        Returns:
            float: 각도 (도 단위)
        """
        unit_a = a / (np.linalg.norm(a) + 1e-8)
        unit_b = b / (np.linalg.norm(b) + 1e-8)
        dot = np.clip(np.dot(unit_a, unit_b), -1.0, 1.0)
        angle_rad = np.arccos(dot)
        return np.degrees(angle_rad)
    
    def extract_angles_from_frame(self, frame):
        """
        프레임에서 4개 핵심 각도 추출
        Args:
            frame: OpenCV 프레임
        Returns:
            tuple: (처리된 프레임, 각도 리스트 또는 None)
        """
        # BGR을 RGB로 변환
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        
        # 포즈 감지
        results = self.pose.process(rgb_frame)
        
        # 다시 BGR로 변환
        rgb_frame.flags.writeable = True
        processed_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        
        angles = None
        
        # 랜드마크가 감지된 경우 각도 계산
        if results.pose_landmarks:
            angles = self._calculate_four_angles(results.pose_landmarks.landmark)
            
            # 포즈 랜드마크 그리기
            self.mp_drawing.draw_landmarks(
                processed_frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )
        
        return processed_frame, angles
    
    def _calculate_four_angles(self, landmarks):
        """
        MediaPipe 랜드마크에서 4개 핵심 각도 계산
        Args:
            landmarks: MediaPipe pose landmarks
        Returns:
            list: [left_elbow, right_elbow, left_knee, right_knee] 또는 None
        """
        try:
            # MediaPipe 랜드마크를 3D 좌표 배열로 변환
            points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
            
            # 각 관절의 3D 좌표 추출
            ls = points[self.joint_indices["left_shoulder"]]
            le = points[self.joint_indices["left_elbow"]]
            lw = points[self.joint_indices["left_wrist"]]
            rs = points[self.joint_indices["right_shoulder"]]
            re = points[self.joint_indices["right_elbow"]]
            rw = points[self.joint_indices["right_wrist"]]
            lh = points[self.joint_indices["left_hip"]]
            lk = points[self.joint_indices["left_knee"]]
            la = points[self.joint_indices["left_ankle"]]
            rh = points[self.joint_indices["right_hip"]]
            rk = points[self.joint_indices["right_knee"]]
            ra = points[self.joint_indices["right_ankle"]]
            
            # 4개 각도 계산 (기준 코드와 완전히 동일한 공식)
            left_elbow_angle = self.calculate_angle(le - ls, lw - le)
            right_elbow_angle = self.calculate_angle(re - rs, rw - re)
            left_knee_angle = self.calculate_angle(lk - lh, la - lk)
            right_knee_angle = self.calculate_angle(rk - rh, ra - rk)
            
            return [left_elbow_angle, right_elbow_angle, left_knee_angle, right_knee_angle]
            
        except Exception:
            return None
    
    def add_frame_data(self, angles, timestamp=None):
        """
        프레임별 각도 데이터 추가 (메인 시스템에서 호출)
        Args:
            angles: [left_elbow, right_elbow, left_knee, right_knee]
            timestamp: 타임스탬프 (None이면 현재 시간)
        """
        if angles is None or len(angles) != 4:
            return
        
        if timestamp is None:
            timestamp = time.time()
        
        self.recorded_data.append({
            'timestamp': timestamp,
            'left_elbow': angles[0],
            'right_elbow': angles[1],
            'left_knee': angles[2],
            'right_knee': angles[3]
        })
    
    def get_angles_as_dataframe(self):
        """
        저장된 각도 데이터를 DataFrame으로 반환 (기준 데이터와 동일한 형식)
        Returns:
            pandas.DataFrame: 각도 데이터 또는 None
        """
        if not self.recorded_data:
            return None
        
        # DataFrame 생성 (기준 데이터와 동일한 컬럼명)
        df_data = []
        for record in self.recorded_data:
            df_data.append([
                record['left_elbow'],
                record['right_elbow'],
                record['left_knee'],
                record['right_knee']
            ])
        
        df = pd.DataFrame(df_data, columns=['Left Elbow', 'Right Elbow', 'Left Knee', 'Right Knee'])
        return df
    
    def get_angles_as_numpy(self):
        """
        저장된 각도 데이터를 NumPy 배열로 반환 (DTW 알고리즘용)
        Returns:
            numpy.ndarray: (N, 4) 형태의 각도 데이터 또는 None
        """
        df = self.get_angles_as_dataframe()
        return df.values if df is not None else None
                
    def get_data_info(self):
        """
        저장된 데이터 정보 반환
        Returns:
            dict: 데이터 정보
        """
        if not self.recorded_data:
            return {'frame_count': 0, 'duration': 0}
        
        start_time = self.recorded_data[0]['timestamp']
        end_time = self.recorded_data[-1]['timestamp']
        duration = end_time - start_time
        
        return {
            'frame_count': len(self.recorded_data),
            'duration': duration,
            'fps': len(self.recorded_data) / duration if duration > 0 else 0
        }
    
    def clear_data(self):
        """저장된 데이터 초기화"""
        self.recorded_data = []
    
    def cleanup(self):
        """리소스 정리"""
        if hasattr(self, 'pose'):
            self.pose.close()

# 테스트 함수 (메인 시스템에서는 사용하지 않음)
def test_angle_calculator():
    """각도 계산기 단독 테스트"""
    print("=== 각도 계산기 단독 테스트 ===")
    print("이 테스트는 메인 시스템과 별도로 각도 계산 기능만 확인합니다.")
    print("조작법: Q키로 종료")
    print("-" * 40)
    
    calculator = AngleCalculator()
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        processed_frame, angles = calculator.extract_angles_from_frame(frame)
        
        # 각도 정보 표시
        if angles:
            y_pos = 30
            for i, (name, angle) in enumerate(zip(['LE', 'RE', 'LK', 'RK'], angles)):
                cv2.putText(processed_frame, f"{name}: {angle:.1f}°", 
                           (10, y_pos + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 테스트용 데이터 추가
            calculator.add_frame_data(angles)
        
        # 데이터 정보 표시
        info = calculator.get_data_info()
        cv2.putText(processed_frame, f"Frames: {info['frame_count']}", 
                   (processed_frame.shape[1]-150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        cv2.imshow('Angle Calculator Test', processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    calculator.cleanup()
    
    # 테스트 결과
    print(f"테스트 완료: {info['frame_count']}프레임 수집됨")
    return calculator

if __name__ == "__main__":
    test_angle_calculator()