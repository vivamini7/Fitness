import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean

class DTWMatcher:
    """메인 시스템에서 사용할 DTW 매칭 전용 클래스"""
    
    def __init__(self, reference_csv_path="candy_angles.csv"):
        """
        DTW 매처 초기화
        Args:
            reference_csv_path: 기준 댄스 각도 데이터 CSV 파일 경로
        """
        try:
            self.reference_data = pd.read_csv(reference_csv_path)
            self.reference_angles = self.reference_data[['Left Elbow', 'Right Elbow', 'Left Knee', 'Right Knee']].values
        except FileNotFoundError:
            raise FileNotFoundError(f"기준 데이터 파일을 찾을 수 없습니다: {reference_csv_path}")
        
        # DTW 결과 저장
        self.dtw_matrix = None
        self.alignment_path = None
        self.dtw_distance = None
    
    def align_sequences(self, user_angles_numpy):
        """
        사용자 댄스와 기준 댄스를 DTW로 정렬 (메인 시스템용)
        Args:
            user_angles_numpy: 사용자 각도 데이터 (N x 4 NumPy 배열)
        Returns:
            tuple: (정렬된_기준_데이터, 정렬된_사용자_데이터) 또는 (None, None)
        """
        if user_angles_numpy is None or len(user_angles_numpy) == 0:
            return None, None
        
        # DTW 매트릭스 계산
        self.dtw_distance = self._calculate_dtw_matrix(user_angles_numpy)
        
        # 최적 정렬 경로 추출
        self.alignment_path = self._get_alignment_path(user_angles_numpy)
        
        # 정렬된 시퀀스 생성
        aligned_reference, aligned_user = self._create_aligned_sequences(user_angles_numpy)
        
        return aligned_reference, aligned_user
    
    def _calculate_dtw_matrix(self, user_angles):
        """DTW 매트릭스 계산"""
        ref_seq = self.reference_angles
        user_seq = user_angles
        
        n, m = len(ref_seq), len(user_seq)
        
        # DTW 매트릭스 초기화
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0
        
        # 동적 프로그래밍으로 최소 비용 경로 계산
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                # 4개 각도의 유클리드 거리
                cost = euclidean(ref_seq[i-1], user_seq[j-1])
                
                # 최소 비용 경로 선택
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i-1, j],     # 기준 프레임 건너뛰기
                    dtw_matrix[i, j-1],     # 사용자 프레임 건너뛰기
                    dtw_matrix[i-1, j-1]    # 1:1 매칭
                )
        
        self.dtw_matrix = dtw_matrix
        return dtw_matrix[n, m]
    
    def _get_alignment_path(self, user_angles):
        """백트래킹으로 최적 정렬 경로 추출"""
        if self.dtw_matrix is None:
            return None
        
        n, m = len(self.reference_angles), len(user_angles)
        path = []
        i, j = n, m
        
        # 백트래킹
        while i > 0 and j > 0:
            path.append((i-1, j-1))  # 0-based 인덱스
            
            # 최소 비용 방향 선택
            diagonal = self.dtw_matrix[i-1, j-1]
            left = self.dtw_matrix[i, j-1]
            up = self.dtw_matrix[i-1, j]
            
            if diagonal <= left and diagonal <= up:
                i, j = i-1, j-1
            elif up <= left:
                i = i-1
            else:
                j = j-1
        
        return list(reversed(path))
    
    def _create_aligned_sequences(self, user_angles):
        """정렬 경로에 따라 정렬된 시퀀스 생성"""
        if self.alignment_path is None:
            return None, None
        
        aligned_reference = []
        aligned_user = []
        
        for ref_idx, user_idx in self.alignment_path:
            aligned_reference.append(self.reference_angles[ref_idx])
            aligned_user.append(user_angles[user_idx])
        
        return np.array(aligned_reference), np.array(aligned_user)
    
    def get_alignment_info(self):
        """정렬 결과 정보 반환"""
        if self.alignment_path is None:
            return None
        
        return {
            'dtw_distance': self.dtw_distance,
            'aligned_length': len(self.alignment_path),
            'reference_length': len(self.reference_angles),
            'compression_ratio': len(self.alignment_path) / max(len(self.reference_angles), len(self.alignment_path))
        }

# 테스트 함수 (메인 시스템에서는 사용하지 않음)
def test_dtw_matcher():
    """DTW 매처 단독 테스트"""
    print("=== DTW 매처 단독 테스트 ===")
    print("이 테스트는 메인 시스템과 별도로 DTW 기능만 확인합니다.")
    
    try:
        matcher = DTWMatcher("candy_angles.csv")
        print(f"✅ 기준 데이터 로드: {len(matcher.reference_angles)}프레임")
    except FileNotFoundError:
        print("❌ candy_angles.csv 파일이 필요합니다.")
        return None
    
    # 테스트용 가상 사용자 데이터 생성
    print("\n🧪 가상 사용자 데이터로 테스트:")
    
    # 기준 데이터 기반으로 노이즈가 있는 가상 데이터 생성
    np.random.seed(42)
    noise = np.random.normal(0, 5, matcher.reference_angles.shape)  # 5도 표준편차 노이즈
    fake_user_data = matcher.reference_angles + noise
    
    # 길이도 조금 다르게 (시간 차이 시뮬레이션)
    fake_user_data = fake_user_data[::2]  # 절반 길이로 (빠른 댄스 시뮬레이션)
    
    print(f"   기준 데이터: {len(matcher.reference_angles)}프레임")
    print(f"   가상 사용자 데이터: {len(fake_user_data)}프레임")
    
    # DTW 정렬 수행
    aligned_ref, aligned_user = matcher.align_sequences(fake_user_data)
    
    if aligned_ref is not None:
        info = matcher.get_alignment_info()
        print(f"\n✅ DTW 정렬 완료:")
        print(f"   DTW 거리: {info['dtw_distance']:.2f}")
        print(f"   정렬된 길이: {info['aligned_length']}프레임")
        print(f"   압축률: {info['compression_ratio']:.2f}")
        
        # 정렬 품질 확인
        frame_differences = []
        for i in range(len(aligned_ref)):
            diff = np.linalg.norm(aligned_ref[i] - aligned_user[i])
            frame_differences.append(diff)
        
        print(f"\n📊 정렬 품질:")
        print(f"   평균 프레임 차이: {np.mean(frame_differences):.2f}")
        print(f"   최대 프레임 차이: {np.max(frame_differences):.2f}")
        print(f"   최소 프레임 차이: {np.min(frame_differences):.2f}")
        
        return matcher, aligned_ref, aligned_user
    else:
        print("❌ DTW 정렬에 실패했습니다.")
        return None

if __name__ == "__main__":
    test_dtw_matcher()