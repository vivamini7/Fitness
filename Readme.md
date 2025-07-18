# Fitness

스쿼트 자세 분석기 (FastAPI + React)

## 백엔드 실행 방법

1. Python 3.8+ 설치
2. 필요한 패키지 설치:
   ```
   pip install fastapi uvicorn mediapipe opencv-python numpy
   ```
3. 서버 실행:
   ```
   cd Backend
   uvicorn main:app --reload
   ```

## 프론트엔드 실행 방법

1. Node.js 설치
2. 프론트엔드 폴더로 이동:
   ```
   cd Frontend
   ```
3. 패키지 설치:
   ```
   npm install
   ```
4. 실행:
   ```
   npm start
   ```

## 주요 기능

- 웹캠으로 스쿼트 자세 측정
- 5회 측정 후 점수 및 랭킹 제공

## 폴더 구조

- Backend: FastAPI 서버
- Frontend/my: React 클라이언트
