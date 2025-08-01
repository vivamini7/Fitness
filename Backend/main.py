from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import mediapipe as mp
import numpy as np
import cv2
import base64
import time
import json
import os

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

# ✅ 랭킹 파일 설정
RANKING_FILE = "rankings.json"
if os.path.exists(RANKING_FILE):
    with open(RANKING_FILE, "r") as f:
        persistent_rankings = json.load(f)
else:
    persistent_rankings = []

def save_rankings():
    with open(RANKING_FILE, "w", encoding="utf-8") as f:
        json.dump(persistent_rankings, f, indent=2, ensure_ascii=False)

reference_stats = {
    "trunk_angle": {"mean": 32.59},
    "hip_angle": {"mean": 110.11},
    "knee_angle": {"mean": 108.35},
    "knee_valgus_dist": {"mean": 49.13}
}

REQUIRED_LANDMARKS = [11, 23, 25, 26, 27]

def are_required_landmarks_present(landmarks, indices):
    return all(landmarks[i].visibility > 0.5 for i in indices)

def compute_score(value, mean):
    diff = abs(value - mean)
    return max(10, 20 - diff + 10)

def get_cycle_label(score):
    if score < 13:
        return "😢 Bad"
    elif score < 15:
        return "🙂 Normal"
    elif score < 18:
        return "👍 Good"
    else:
        return "🏅 Perfect"

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return np.degrees(angle)

def draw_overlay(image, status, angle_records, cycle_scores):
    cv2.putText(image, status, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    for i, score in enumerate(cycle_scores):
        text = f"{i+1} cycle: {score:.1f}점 ({get_cycle_label(score)})"
        x, y = image.shape[1] - 400, 50 + i * 40
        cv2.rectangle(image, (x-10, y-30), (x+350, y), (255, 255, 255), -1)
        cv2.putText(image, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    return image

@app.post("/analyze")
async def analyze_pose(image: UploadFile = File(...), user_id: str = Form(...)):
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
            "angle_buffer": [],
            "total_score": None,
            "cycle_scores": []
        }

    session = user_sessions[user_id]
    current_time = time.time()
    status = "⏳ 분석 중..."
    cycle_score = None

    if results.pose_landmarks and are_required_landmarks_present(results.pose_landmarks.landmark, REQUIRED_LANDMARKS):
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

    if session["cycle_count"] >= 3 and session.get("total_score") is None:
        status = "✅ 3회 측정 완료"
        avg_record = {
            key: np.mean([r[key] for r in session["records"]])
            for key in session["records"][0]
        }
        score_record = {
            key: compute_score(avg_record[key], reference_stats[key]["mean"])
            for key in avg_record
        }
        session["total_score"] = round(sum(score_record.values()), 2)

        existing = next((r for r in persistent_rankings if r["user_id"] == user_id), None)
        if existing:
            existing["total_score"] = session["total_score"]
        else:
            persistent_rankings.append({
                "user_id": user_id,
                "total_score": session["total_score"]
            })
        save_rankings()

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
                    score = np.mean([
                        compute_score(avg[k], reference_stats[k]["mean"])
                        for k in avg
                    ])
                    session["cycle_scores"].append(round(score, 2))
                    cycle_score = round(score, 2)

                session["cycle_count"] += 1

                if session["cycle_count"] == 3 and session.get("total_score") is None:
                    avg_record = {
                        key: np.mean([r[key] for r in session["records"]])
                        for key in session["records"][0]
                    }
                    score_record = {
                        key: compute_score(avg_record[key], reference_stats[key]["mean"])
                        for key in avg_record
                    }
                    session["total_score"] = round(sum(score_record.values()), 2)

                    existing = next((r for r in persistent_rankings if r["user_id"] == user_id), None)
                    if existing:
                        existing["total_score"] = session["total_score"]
                    else:
                        persistent_rankings.append({
                            "user_id": user_id,
                            "total_score": session["total_score"]
                        })
                    save_rankings()

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

    annotated = draw_overlay(img, status, session["records"], session["cycle_scores"])
    _, buffer = cv2.imencode(".jpg", annotated)
    encoded_img = base64.b64encode(buffer.tobytes()).decode('utf-8')

    return JSONResponse({
        "image_base64": encoded_img,
        "status": status,
        "cycle_score": cycle_score,
        "cycle_label": get_cycle_label(cycle_score) if cycle_score else None,
        "total_score": session.get("total_score"),
        "cycle_scores": session["cycle_scores"]
    })

@app.get("/ranking")
async def get_ranking():
    sorted_rankings = sorted(persistent_rankings, key=lambda x: x["total_score"], reverse=True)
    return JSONResponse({"ranking": sorted_rankings})

@app.post("/ranking/reset")
async def reset_ranking():
    global persistent_rankings
    persistent_rankings = []
    save_rankings()
    return JSONResponse({"message": "랭킹 초기화 완료"})

class SaveRequest(BaseModel):
    user_id: str
    total_score: float

@app.post("/ranking/save")
async def save_score(req: SaveRequest):
    existing = next((r for r in persistent_rankings if r["user_id"] == req.user_id), None)
    if existing:
        existing["total_score"] = req.total_score
    else:
        persistent_rankings.append({
            "user_id": req.user_id,
            "total_score": req.total_score
        })
    save_rankings()
    return JSONResponse({"message": "저장 완료"})
