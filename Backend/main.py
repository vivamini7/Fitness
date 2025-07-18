from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import mediapipe as mp
import numpy as np
import cv2
import base64
import time

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pose = mp.solutions.pose.Pose(static_image_mode=True)
user_sessions = {}

reference_stats = {
    "trunk_angle": {"mean": 32.59},
    "hip_angle": {"mean": 110.11},
    "knee_angle": {"mean": 108.35},
    "knee_valgus_dist": {"mean": 49.13}
}

def compute_score(value, mean):
    diff = abs(value - mean)
    return max(10, 25 - diff)

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return np.degrees(angle)

def draw_overlay(image, status, angle_records):
    cv2.putText(image, status, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    for i, record in enumerate(angle_records):
        text = f"{i+1} cycle: {record['knee_angle']:.2f}°"
        x, y = image.shape[1] - 250, 50 + i * 40
        cv2.rectangle(image, (x-10, y-30), (x+220, y), (255, 255, 255), -1)
        cv2.putText(image, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    return image

@app.post("/analyze")
async def analyze_pose(image: UploadFile = File(...), user_id: str = Form(...)):
    global user_sessions

    content = await image.read()
    np_img = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    if user_id not in user_sessions:
        user_sessions[user_id] = {
            "records": [],
            "state": "standing",
            "state_timer": None,
            "cycle_count": 0,
            "angle_buffer": []
        }

    session = user_sessions[user_id]
    current_time = time.time()
    status = "⏳ 분석 중..."
    score_record = {}
    total_score = None

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        shoulder = [lm[11].x, lm[11].y]
        hip = [lm[23].x, lm[23].y]
        knee = [lm[25].x, lm[25].y]
        ankle = [lm[27].x, lm[27].y]
        l_knee = lm[25]
        r_knee = lm[26]

        trunk_angle = calculate_angle(shoulder, hip, knee)
        hip_angle = trunk_angle
        knee_angle = calculate_angle(hip, knee, ankle)
        knee_valgus_dist = abs(l_knee.x - r_knee.x) * img.shape[1]

        angles_this_frame = {
            "trunk_angle": trunk_angle,
            "hip_angle": hip_angle,
            "knee_angle": knee_angle,
            "knee_valgus_dist": knee_valgus_dist
        }
    else:
        angles_this_frame = {}

    if session["cycle_count"] >= 5:
        status = "✅ 5회 측정 완료"

        avg_record = {
            key: np.mean([r[key] for r in session["records"]])
            for key in session["records"][0]
        }

        score_record = {
            key: compute_score(avg_record[key], reference_stats[key]["mean"])
            for key in avg_record
        }

        total_score = round(sum(score_record.values()), 2)
        session["total_score"] = total_score

        print(f"[{user_id}] 평균 측정값: {avg_record}")
        print(f"[{user_id}] 점수: {score_record}, 총점: {total_score}")

    else:
        if session["state_timer"] is None:
            session["state_timer"] = current_time

        elapsed = current_time - session["state_timer"]

        if session["state"] == "standing":
            status = "🧍 준비하세요"
            if elapsed >= 3:
                session["state"] = "descending"
                session["state_timer"] = current_time

        elif session["state"] == "descending":
            status = "⬇️ 내려가세요"
            if elapsed >= 2:
                session["state"] = "hold"
                session["state_timer"] = current_time
                session["angle_buffer"] = []

        elif session["state"] == "hold":
            status = "⏱️ 유지 중..."
            if angles_this_frame:
                session["angle_buffer"].append(angles_this_frame)
            if elapsed >= 2:
                if session["angle_buffer"]:
                    avg = {
                        key: np.mean([a[key] for a in session["angle_buffer"]])
                        for key in session["angle_buffer"][0]
                    }
                    session["records"].append(avg)
                session["cycle_count"] += 1
                session["state"] = "ascending"
                session["state_timer"] = current_time

        elif session["state"] == "ascending":
            status = "⬆️ 올라오세요"
            if elapsed >= 2:
                session["state"] = "rest"
                session["state_timer"] = current_time

        elif session["state"] == "rest":
            status = "😮‍💨 휴식 중..."
            if elapsed >= 3:
                session["state"] = "standing"
                session["state_timer"] = current_time

    annotated = draw_overlay(img, status, session["records"])
    _, buffer = cv2.imencode(".jpg", annotated)
    encoded_img = base64.b64encode(buffer.tobytes()).decode('utf-8')

    return JSONResponse({
        "image_base64": encoded_img,
        "status": status,
        "angles": session["records"],
        "score_detail": score_record,
        "total_score": total_score
    })

@app.get("/ranking")
async def get_ranking():
    rankings = []
    for user_id, session in user_sessions.items():
        if "total_score" in session and len(session["records"]) == 5:
            rankings.append({
                "user_id": user_id,
                "total_score": session["total_score"]
            })

    rankings.sort(key=lambda x: x["total_score"], reverse=True)
    return JSONResponse({"ranking": rankings})
