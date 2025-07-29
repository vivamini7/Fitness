import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class AngleVerifier:
    """각도 계산 결과를 기준 데이터와 비교하여 검증하는 클래스"""
    
    def __init__(self, reference_csv_path="candy_angles.csv"):
        """
        Args:
            reference_csv_path: 기준 각도 데이터 CSV 파일 경로
        """
        try:
            self.reference_data = pd.read_csv(reference_csv_path)
            print(f"✅ 기준 데이터 로드 완료: {len(self.reference_data)}프레임")
        except FileNotFoundError:
            print(f"❌ 기준 데이터 파일을 찾을 수 없습니다: {reference_csv_path}")
            self.reference_data = None
    

    
    def compare_with_reference(self, user_csv_path):
        """사용자 데이터를 기준 데이터와 비교 (DTW 방식으로)"""
        if self.reference_data is None:
            print("기준 데이터가 없습니다.")
            return
        
        try:
            user_data = pd.read_csv(user_csv_path)
            print(f"\n✅ 사용자 데이터 로드 완료: {len(user_data)}프레임")
        except FileNotFoundError:
            print(f"❌ 사용자 데이터 파일을 찾을 수 없습니다: {user_csv_path}")
            return
        
        # 데이터 비교 (DTW 매칭 방식 시뮬레이션)
        print("\n📈 프레임별 비교 분석 (DTW 방식):")
        print(f"   기준 데이터: {len(self.reference_data)}프레임")
        print(f"   사용자 데이터: {len(user_data)}프레임")
        
        # 간단한 DTW 시뮬레이션 (길이 조정)
        ref_angles = self.reference_data[['Left Elbow', 'Right Elbow', 'Left Knee', 'Right Knee']].values
        user_angles = user_data[['Left Elbow', 'Right Elbow', 'Left Knee', 'Right Knee']].values
        
        # 더 짧은 길이에 맞춰 리샘플링 (간단한 DTW 대용)
        min_length = min(len(ref_angles), len(user_angles))
        ref_resampled = self._resample_data(ref_angles, min_length)
        user_resampled = self._resample_data(user_angles, min_length)
        
        print(f"   매칭된 프레임 수: {min_length}")
        
        # 프레임별 각도 차이 계산
        frame_scores = []
        angle_names = ['Left Elbow', 'Right Elbow', 'Left Knee', 'Right Knee']
        
        for frame_idx in range(min_length):
            frame_diff = []
            for angle_idx in range(4):
                ref_angle = ref_resampled[frame_idx, angle_idx]
                user_angle = user_resampled[frame_idx, angle_idx]
                angle_diff = abs(ref_angle - user_angle)
                frame_diff.append(angle_diff)
            
            # 프레임별 평균 차이
            avg_diff = np.mean(frame_diff)
            frame_scores.append(avg_diff)
        
        # 전체 통계
        total_avg_diff = np.mean(frame_scores)
        print(f"\n📊 프레임별 각도 차이 통계:")
        print(f"   전체 평균 차이: {total_avg_diff:.1f}°")
        print(f"   최소 차이: {min(frame_scores):.1f}°")
        print(f"   최대 차이: {max(frame_scores):.1f}°")
        print(f"   표준편차: {np.std(frame_scores):.1f}°")
        
        # 각도별 평균 차이
        print(f"\n📈 각도별 평균 차이:")
        for angle_idx, angle_name in enumerate(angle_names):
            angle_diffs = []
            for frame_idx in range(min_length):
                ref_angle = ref_resampled[frame_idx, angle_idx]
                user_angle = user_resampled[frame_idx, angle_idx]
                angle_diffs.append(abs(ref_angle - user_angle))
            
            avg_diff = np.mean(angle_diffs)
            print(f"   {angle_name}: {avg_diff:.1f}°")
            
            # 유사도 판정
            if avg_diff < 15:
                print(f"     ✅ 매우 유사함")
            elif avg_diff < 25:
                print(f"     ⚠️ 보통 유사함")
            else:
                print(f"     ❌ 차이가 큼")
        
        return user_data, frame_scores
    
    def _resample_data(self, data, target_length):
        """데이터를 목표 길이로 리샘플링 (간단한 선형 보간)"""
        if len(data) == target_length:
            return data
        
        # 선형 보간으로 리샘플링
        original_indices = np.linspace(0, len(data) - 1, len(data))
        target_indices = np.linspace(0, len(data) - 1, target_length)
        
        resampled = np.zeros((target_length, data.shape[1]))
        for col in range(data.shape[1]):
            resampled[:, col] = np.interp(target_indices, original_indices, data[:, col])
        
        return resampled
    

    


def quick_test():
    """빠른 테스트 함수 (프레임별 비교 방식)"""
    print("=== 각도 계산 검증 도구 (프레임별 비교) ===")
    
    # 검증기 생성
    verifier = AngleVerifier("candy_angles.csv")
    
    # 사용자 데이터가 있다면 비교
    import os
    user_files = [f for f in os.listdir('.') if f.startswith('user_angles_') and f.endswith('.csv')]
    
    if user_files:
        print(f"\n📁 발견된 사용자 데이터 파일: {user_files}")
        latest_file = max(user_files)  # 가장 최근 파일
        print(f"🔍 가장 최근 파일로 분석: {latest_file}")
        
        # 프레임별 비교 분석
        result = verifier.compare_with_reference(latest_file)
        if result is not None:
            user_data, frame_scores = result
            
            # 점수 예측 (실제 DTW + 점수 계산 미리보기)
            print(f"\n🎯 예상 최종 점수 미리보기:")
            
            # 간단한 점수 계산 (15도 차이를 기준으로)
            score_preview = []
            for diff in frame_scores:
                if diff <= 15:
                    score = 100 * np.exp(-(diff**2) / (2 * (7.5**2)))  # 가우시안
                else:
                    score = max(0, 20 * (1 - (diff - 15) / 165))  # 선형 감소
                score_preview.append(score)
            
            final_score = np.mean(score_preview)
            print(f"   예상 총점: {final_score:.1f}/100")
            print(f"   프레임별 점수 범위: {min(score_preview):.1f} ~ {max(score_preview):.1f}")
            
            # 등급 예측
            if final_score >= 80:
                print(f"   예상 등급: ⭐ Excellent")
            elif final_score >= 60:
                print(f"   예상 등급: 👍 Good")
            elif final_score >= 40:
                print(f"   예상 등급: ⚠️ Fair")
            else:
                print(f"   예상 등급: ❌ Needs Practice")
    else:
        print("\n📝 사용자 각도 데이터가 없습니다.")
        print("   먼저 AngleCalculator.run_test()로 데이터를 수집하세요.")
        print("   🎯 DTW 매칭 후 프레임별로 4개 각도를 비교하여 점수를 계산할 예정입니다.")

if __name__ == "__main__":
    quick_test()