from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
import json
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS 허용 설정 (React와 통신할 수 있도록)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_FILE = Path("rankings_hard.json")

class ScoreEntry(BaseModel):
    user_id: str
    total_score: int

@app.get("/api/hard-ranking")
def get_hard_ranking():
    if DATA_FILE.exists():
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

@app.post("/api/hard-ranking")
def add_hard_ranking(entry: ScoreEntry):
    data = []
    if DATA_FILE.exists():
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

    data.append(entry.dict())

    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return {"message": "저장 완료"}
