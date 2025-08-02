# Fitness

**스쿼트 자세 분석기**  
웹캠 기반 자세 평가 & 랭킹 시스템  
기술 스택: `FastAPI` + `React` + `MediaPipe`

| 팀원 | GitHub                           |
| --------- | ------------------------------------- |
| 🐿️유지혜    | [github.com/jihyeyu33](https://github.com/jihyeyu33)       |
| 🐼윤상민    | [github.com/vivamini7](https://github.com/vivamini7) |
| 🦭최서연    | [github.com/seoyeon-eo](https://github.com/seoyeon-eo)         |
| 🦢한예원    | [github.com/LoLe05](https://github.com/LoLe05)       |

## 폴더 구조
```
📦 Fitness
 ┣ 📂 Backend              # FastAPI 기반 백엔드
 ┃ ┣ main.py              # Easy 모드 서버
 ┃ ┣ main_hard.py         # Hard 모드 서버
 ┃ ┗ 📂 dance_demo
 ┃   ┗ hard_main.py       # 댄스 모드 서버
 ┣ 📂 Frontend/my         # React 기반 프론트엔드
 ┃ ┗ App.js, index.js, ...
 ┗ README.md              # 설명 문서
```

## 백엔드 실행 방법

1. 필요한 패키지 설치:
   ```
   cd Backend
   pip install requirement.txt
   ```
2. 서버 실행 easy_mode:
   ```
   cd Backend
   uvicorn main:app --reload
   ```
3. 서버 실행 hard_mode1 :
   ```
   cd Backend
   uvicorn main_hard:app --reload
   ```
4. 서버 실행 hard_mode2 :
   ```
   cd Backend
   cd dance_demo
   uvicorn hard_main:app --reload
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
✅ 웹캠을 활용한 스쿼트 자세 실시간 분석

✅ 총 3회 측정 후 자동 점수 산정

✅ 사용자별 측정 기록 랭킹 제공

✅ 하드모드에서 candy dance에 대해 평가

## 📄 프로젝트 소개 패널

아래 PDF는 본 프로젝트의 개요, 기술 스택, 동작 흐름, Easy/Hard 모드 구현 내용을 시각적으로 정리한 발표용 패널입니다.

👉 [5팀_판넬.pdf 다운로드 및 보기](./5팀_판넬.pdf)
