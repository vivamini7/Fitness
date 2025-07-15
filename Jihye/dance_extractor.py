import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mediapipe as mp
import os


# 필수 패키지 목록:
# pip install yt-dlp mediapipe==0.10.9 opencv-python matplotlib numpy

def download_youtube_video(url, output_path="dance_video.mp4"):
    """유튜브 영상 다운로드"""
    try:
        import yt_dlp
        
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
            'merge_output_format': 'mp4',
            'outtmpl': output_path,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print(f"🎥 유튜브 영상 다운로드 중: {url}")
            ydl.download([url])
            print(f"✅ 다운로드 완료: {output_path}")
            return output_path
            
    except Exception as e:
        print(f"❌ 다운로드 실패: {e}")
        return None

def get_video_info(video_path):
    """영상 정보 확인"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 영상을 열 수 없습니다: {video_path}")
        return None
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    
    cap.release()
    
    info = {
        'fps': fps,
        'frame_count': frame_count,
        'duration': duration
    }
    
    print(f"🎥 영상 프레임 수: {frame_count}")
    print(f"🎞 FPS: {fps}")
    print(f"⏱ 영상 길이 (초): {duration:.2f}")
    
    return info

def extract_pose_landmarks(video_path, start_time=0, end_time=30, target_fps=15):
    """영상에서 포즈 랜드마크 추출"""
    
    # MediaPipe 초기화
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # 영상 열기
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 영상을 열 수 없습니다: {video_path}")
        return None
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    # 실제 종료 시간 조정
    end_time = min(end_time, duration)
    
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    step = max(1, int(fps / target_fps))
    
    print(f"🎬 분석 구간: {start_time}초 ~ {end_time:.1f}초")
    print(f"📊 목표 FPS: {target_fps}")
    print(f"🔄 처리할 프레임: {start_frame} ~ {end_frame} (간격: {step})")
    
    # 포즈 데이터 추출
    landmark_list = []
    frame_idx = 0
    processed_frames = 0
    
    print("📍 포즈 추출 중...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_idx > end_frame:
            break
            
        if frame_idx >= start_frame and (frame_idx - start_frame) % step == 0:
            # BGR → RGB 변환
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            
            if results.pose_landmarks:
                # 33개 랜드마크의 x, y, z 좌표 추출
                landmarks = [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]
                landmark_list.append(landmarks)
                processed_frames += 1
                
                # 진행상황 출력
                if processed_frames % 50 == 0:
                    print(f"  처리된 프레임: {processed_frames}")
        
        frame_idx += 1
    
    cap.release()
    pose.close()
    
    if len(landmark_list) > 0:
        print(f"✅ 포즈 추출 완료! 총 {len(landmark_list)}개 프레임 처리")
        landmark_array = np.array(landmark_list)
        print(f"📊 데이터 형태: {landmark_array.shape}")
        return landmark_array
    else:
        print("❌ 포즈를 찾을 수 없습니다. 영상에 사람이 명확히 보이는지 확인하세요.")
        return None

def create_pose_animation(landmark_data, target_fps=15, save_gif=False, gif_filename="dance_animation.gif"):
    """포즈 데이터로 애니메이션 생성"""
    
    if landmark_data is None or len(landmark_data) == 0:
        print("❌ 애니메이션할 데이터가 없습니다.")
        return
    
    print("🎬 애니메이션 생성 중...")
    
    # MediaPipe 연결 정보
    mp_pose = mp.solutions.pose
    connections = mp_pose.POSE_CONNECTIONS
    
    # 플롯 설정
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 그래픽 요소 초기화
    points, = ax.plot([], [], 'ro', markersize=6, alpha=0.8)
    lines = [ax.plot([], [], 'b-', linewidth=2, alpha=0.7)[0] for _ in connections]
    
    def init():
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)  # y축 반전
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_facecolor('black')  # 배경을 검정으로
        return [points] + lines
    
    def update(frame_idx):
        if frame_idx < len(landmark_data):
            landmarks = landmark_data[frame_idx]
            xs = [lm[0] for lm in landmarks]
            ys = [lm[1] for lm in landmarks]
            
            # 관절점 업데이트
            points.set_data(xs, ys)
            
            # 연결선 업데이트
            for i, (start, end) in enumerate(connections):
                x_line = [landmarks[start][0], landmarks[end][0]]
                y_line = [landmarks[start][1], landmarks[end][1]]
                lines[i].set_data(x_line, y_line)
            
            # 프레임 정보 표시
            ax.set_title(f'안무 동작 분석 - Frame {frame_idx+1}/{len(landmark_data)}', 
                        color='white', fontsize=16, pad=20)
        
        return [points] + lines
    
    # 애니메이션 생성
    animation_interval = max(30, int(1000 / target_fps))
    ani = animation.FuncAnimation(
        fig, update,
        frames=len(landmark_data),
        init_func=init,
        interval=animation_interval,
        blit=False,
        repeat=True
    )
    
    plt.tight_layout()
    
    # GIF로 저장 (선택사항)
    if save_gif:
        print(f"💾 GIF 저장 중: {gif_filename}")
        ani.save(gif_filename, writer='pillow', fps=target_fps)
        print(f"✅ GIF 저장 완료: {gif_filename}")
    
    # 화면에 표시
    plt.show()
    
    print(f"✅ 애니메이션 완료! {len(landmark_data)}프레임")
    
    return ani

def save_pose_data(landmark_data, filename="dance_pose.npy"):
    """포즈 데이터를 파일로 저장"""
    if landmark_data is not None:
        np.save(filename, landmark_data)
        file_size_kb = landmark_data.nbytes / 1024
        print(f"💾 포즈 데이터 저장 완료: {filename}")
        print(f"   데이터 크기: {file_size_kb:.1f} KB")
        print(f"   형태: {landmark_data.shape}")
        return True
    return False

def main():
    """메인 실행 함수"""
    print("🎵 안무 영상 MediaPipe 분석기")
    print("=" * 40)
    
    
    url = input("유튜브 URL을 입력하세요: ").strip()
    video_path = download_youtube_video(url)
    if not video_path:
        return
    
    # 영상 정보 확인
    print("\n" + "=" * 40)
    video_info = get_video_info(video_path)
    if not video_info:
        return
    
    # 분석 설정
    print("\n" + "=" * 40)
    print("⚙️ 분석 설정")
    
    try:
        start_time = float(input(f"시작 시간 (초, 기본값 0): ") or "0")
        max_duration = min(60, video_info['duration'])  # 최대 60초
        end_time = float(input(f"종료 시간 (초, 기본값 {max_duration:.1f}): ") or str(max_duration))
        target_fps = int(input("목표 FPS (기본값 15): ") or "15")
    except ValueError:
        print("❌ 잘못된 입력값입니다. 기본값을 사용합니다.")
        start_time = 0
        end_time = min(30, video_info['duration'])
        target_fps = 15
    
    # 포즈 추출
    print("\n" + "=" * 40)
    landmark_data = extract_pose_landmarks(video_path, start_time, end_time, target_fps)
    
    if landmark_data is None:
        return
    
    # 데이터 저장 여부
    save_choice = input("\n💾 포즈 데이터를 파일로 저장하시겠습니까? (y/n): ").lower()
    if save_choice == 'y':
        filename = input("파일명 (기본값: dance_pose.npy): ").strip() or "dance_pose.npy"
        save_pose_data(landmark_data, filename)
    
    # 애니메이션 생성
    print("\n" + "=" * 40)
    gif_choice = input("🎬 GIF 파일로도 저장하시겠습니까? (y/n): ").lower()
    save_gif = gif_choice == 'y'
    
    if save_gif:
        gif_filename = input("GIF 파일명 (기본값: dance_animation.gif): ").strip() or "dance_animation.gif"
    else:
        gif_filename = "dance_animation.gif"
    
    # 애니메이션 실행
    create_pose_animation(landmark_data, target_fps, save_gif, gif_filename)
    
    print("\n🎉 모든 작업이 완료되었습니다!")

if __name__ == "__main__":
    main()