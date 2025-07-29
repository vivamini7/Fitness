import numpy as np

class ScoreCalculator:
    """메인 시스템에서 사용할 점수 계산 전용 클래스"""
    
    def __init__(self):
        # 허용 오차 설정
        self.tolerance = {
            'elbow': 15,  # 팔꿈치 ±15°
            'knee': 12    # 무릎 ±12°
        }
        
        # 점수 구간 설정
        self.score_ranges = {
            'elbow': {
                'perfect': (0, 15, 100),      # 0~15°: 100점
                'good': (15, 30, 80),         # 15~30°: 80점
                'fair': (30, 45, 50),         # 30~45°: 50점
                'poor': (45, float('inf'), 20) # 45°+: 20점
            },
            'knee': {
                'perfect': (0, 12, 100),      # 0~12°: 100점
                'good': (12, 24, 80),         # 12~24°: 80점
                'fair': (24, 36, 50),         # 24~36°: 50점
                'poor': (36, float('inf'), 20) # 36°+: 20점
            }
        }
    
    def calculate_final_score(self, aligned_reference, aligned_user):
        """
        DTW 정렬된 데이터로 최종 점수 계산 (메인 시스템용)
        Args:
            aligned_reference: DTW 정렬된 기준 데이터 (N x 4)
            aligned_user: DTW 정렬된 사용자 데이터 (N x 4)
        Returns:
            float: 최종 점수 (0-100) 또는 None
        """
        if aligned_reference is None or aligned_user is None:
            return None
        
        if len(aligned_reference) != len(aligned_user):
            return None
        
        frame_scores = []
        
        # 프레임별 점수 계산
        for i in range(len(aligned_reference)):
            ref_frame = aligned_reference[i]
            user_frame = aligned_user[i]
            
            frame_score = self._calculate_frame_score(ref_frame, user_frame)
            if frame_score is not None:
                frame_scores.append(frame_score)
        
        if not frame_scores:
            return None
        
        # 최종 점수 = 모든 프레임 점수의 평균
        final_score = np.mean(frame_scores)
        return final_score
    
    def _calculate_frame_score(self, ref_angles, user_angles):
        """
        프레임별 점수 계산
        Args:
            ref_angles: 기준 각도 [left_elbow, right_elbow, left_knee, right_knee]
            user_angles: 사용자 각도 [left_elbow, right_elbow, left_knee, right_knee]
        Returns:
            float: 프레임 점수 (0-100)
        """
        if len(ref_angles) != 4 or len(user_angles) != 4:
            return None
        
        # 각도 차이 계산
        angle_diffs = [abs(ref - user) for ref, user in zip(ref_angles, user_angles)]
        
        # 관절 타입 정의
        joint_types = ['elbow', 'elbow', 'knee', 'knee']
        
        # 개별 점수 계산
        individual_scores = []
        for diff, joint_type in zip(angle_diffs, joint_types):
            score = self._get_angle_score(diff, joint_type)
            individual_scores.append(score)
        
        # 프레임 평균 점수 (4개 각도 평균)
        frame_score = sum(individual_scores) / 4
        return frame_score
    
    def _get_angle_score(self, angle_diff, joint_type):
        """
        개별 각도의 점수 계산
        Args:
            angle_diff: 각도 차이 (절댓값)
            joint_type: 'elbow' 또는 'knee'
        Returns:
            int: 점수 (100, 80, 50, 20 중 하나)
        """
        ranges = self.score_ranges[joint_type]
        
        for level, (min_diff, max_diff, score) in ranges.items():
            if min_diff <= angle_diff < max_diff:
                return score
        
        # 기본값
        return 20
    
    def get_score_grade(self, score):
        """
        점수를 등급으로 변환
        Args:
            score: 점수 (0-100)
        Returns:
            str: 등급 문자열
        """
        if score is None:
            return "N/A"
        
        if score >= 90:
            return "S"
        elif score >= 80:
            return "A"
        elif score >= 70:
            return "B"
        elif score >= 60:
            return "C"
        else:
            return "D"

# 테스트 함수 (메인 시스템에서는 사용하지 않음)
def test_score_calculator():
    """점수 계산기 단독 테스트"""
    print("=== 점수 계산기 단독 테스트 ===")
    print("이 테스트는 메인 시스템과 별도로 점수 계산 기능만 확인합니다.")
    
    calculator = ScoreCalculator()
    
    # 테스트 케이스들
    test_cases = [
        {
            'name': 'Perfect Case (완벽한 매칭)',
            'ref': np.array([[90, 95, 110, 105], [85, 100, 115, 100]]),
            'user': np.array([[92, 93, 108, 107], [87, 98, 113, 102]]),
            'expected': '95점 이상'
        },
        {
            'name': 'Good Case (약간 차이)',
            'ref': np.array([[90, 95, 110, 105], [85, 100, 115, 100]]),
            'user': np.array([[105, 80, 125, 90], [100, 85, 130, 85]]),
            'expected': '80점대'
        },
        {
            'name': 'Poor Case (큰 차이)',
            'ref': np.array([[90, 95, 110, 105], [85, 100, 115, 100]]),
            'user': np.array([[140, 50, 160, 60], [135, 55, 155, 65]]),
            'expected': '낮은 점수'
        }
    ]
    
    print(f"\n🧪 점수 계산 로직 테스트:")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n테스트 {i}: {test_case['name']}")
        
        score = calculator.calculate_final_score(test_case['ref'], test_case['user'])
        grade = calculator.get_score_grade(score)
        
        print(f"   최종 점수: {score:.1f}점 (등급: {grade})")
        print(f"   예상: {test_case['expected']}")
        
        # 프레임별 상세 점수
        for j in range(len(test_case['ref'])):
            frame_score = calculator._calculate_frame_score(test_case['ref'][j], test_case['user'][j])
            print(f"   프레임 {j+1}: {frame_score:.1f}점")
    
    # 실제 정렬된 데이터 파일이 있다면 테스트
    print(f"\n🎯 실제 정렬된 데이터로 테스트:")
    
    import os
    import pandas as pd
    
    aligned_files = [f for f in os.listdir('.') if f.startswith('aligned_') and f.endswith('.csv')]
    ref_files = [f for f in aligned_files if 'reference' in f]
    user_files = [f for f in aligned_files if 'user' in f]
    
    if ref_files and user_files:
        latest_ref = max(ref_files)
        latest_user = max(user_files)
        
        print(f"   기준 파일: {latest_ref}")
        print(f"   사용자 파일: {latest_user}")
        
        try:
            ref_data = pd.read_csv(latest_ref).values
            user_data = pd.read_csv(latest_user).values
            
            final_score = calculator.calculate_final_score(ref_data, user_data)
            grade = calculator.get_score_grade(final_score)
            
            print(f"\n🎊 실제 데이터 점수 결과:")
            print(f"   최종 점수: {final_score:.1f}점")
            print(f"   등급: {grade}")
            print(f"   총 프레임: {len(ref_data)}")
            
            return calculator, final_score
            
        except Exception as e:
            print(f"   파일 읽기 오류: {e}")
    else:
        print("   정렬된 데이터 파일이 없습니다.")
        print("   먼저 DTW 알고리즘으로 데이터를 정렬하세요.")
    
    return calculator

if __name__ == "__main__":
    test_score_calculator()