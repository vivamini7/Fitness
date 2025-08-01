import React, { useEffect, useRef, useState, useCallback } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import './App.css';

export default function App() {
  const videoRef = useRef(null);
  const [status, setStatus] = useState("대기 중...");
  const [userId, setUserId] = useState("");
  const [started, setStarted] = useState(false);
  const [cycleScores, setCycleScores] = useState([]);
  const [totalScore, setTotalScore] = useState(null);
  const [latestCycleScore, setLatestCycleScore] = useState(null);
  const [latestLabel, setLatestLabel] = useState(null);
  const [isSending, setIsSending] = useState(false);
  const [overlayLabel, setOverlayLabel] = useState(null);
  const [overlayColor, setOverlayColor] = useState(null);
  const [motionStatus, setMotionStatus] = useState(null);
  const navigate = useNavigate();
  const location = useLocation();
  const queryParams = new URLSearchParams(location.search);
  const mode = queryParams.get("mode") || "easy";

  const getColorByLabel = (label) => {
    if (!label) return "gray";
    if (label.includes("Perfect")) return "limegreen";
    if (label.includes("Good")) return "gold";
    if (label.includes("Normal")) return "orange";
    if (label.includes("Bad")) return "red";
    return "gray";
  };

  const captureAndSend = useCallback(async () => {
    if (isSending || !userId.trim()) return;
    const video = videoRef.current;
    if (!video || video.videoWidth === 0 || video.videoHeight === 0) return;

    setIsSending(true);
    try {
      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', 0.8));
      const formData = new FormData();
      formData.append('image', blob, 'frame.jpg');
      formData.append('user_id', userId);

      const response = await fetch("http://localhost:8000/analyze", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      setStatus(data.status || "⏳ 분석 중...");
      setCycleScores(data.cycle_scores || []);
      setTotalScore(data.total_score ?? null);
      setLatestCycleScore(data.cycle_score || null);
      setLatestLabel(data.cycle_label || null);

      if (
        data.cycle_score &&
        data.cycle_label &&
        data.status?.includes("유지 중")
      ) {
        setOverlayLabel(data.cycle_label);
        setOverlayColor(getColorByLabel(data.cycle_label));
        setTimeout(() => {
          setOverlayLabel(null);
          setOverlayColor(null);
        }, 2000);
      }

      if (data.status && !data.status.includes("분석 중")) {
        setMotionStatus(data.status);
        setTimeout(() => {
          setMotionStatus(null);
        }, 2000);
      }

      if ((data.cycle_scores || []).length >= 3) {  // ✅ 3회 측정 완료 조건
        setStarted(false);
        setStatus("측정 완료! 다시 시작하려면 이름을 바꾸세요.");

        if (data.total_score) {
          await fetch("http://localhost:8000/ranking/save", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ user_id: userId, total_score: data.total_score }),
          });
        }

        setTimeout(() => {
          navigate("/ranking");
        }, 4000);
      }
    } catch (err) {
      console.error("❌ Fetch error:", err);
    } finally {
      setIsSending(false);
    }
  }, [userId, isSending, navigate]);

  useEffect(() => {
    const video = videoRef.current;
    navigator.mediaDevices.getUserMedia({ video: true }).then((stream) => {
      if (!video) return;
      video.srcObject = stream;
      video.onloadedmetadata = () => {
        video.play();
      };
    });
  }, []);

  useEffect(() => {
    if (!started) return;
    const interval = setInterval(() => {
      captureAndSend();
    }, 2000);
    return () => clearInterval(interval);
  }, [started, captureAndSend]);

  const handleStart = () => {
    if (!userId.trim()) {
      alert("이름을 입력해주세요!");
      return;
    }
    setCycleScores([]);
    setTotalScore(null);
    setLatestCycleScore(null);
    setLatestLabel(null);
    setStarted(true);
    setStatus("측정 시작!");
  };

  return (
    <div className="container">
      <h1 className="title">🏋️‍♂️ 스쿼트 자세 분석기</h1>
      <div className="main-content">
        {/* 왼쪽: 카메라 */}
        <div className="left-panel">
          {!started && (
            <div className="start-panel">
              <input
                type="text"
                value={userId}
                placeholder="사용자 이름을 입력하세요"
                onChange={(e) => setUserId(e.target.value)}
              />
              <button onClick={handleStart}>측정 시작</button>
            </div>
          )}

          <div className="video-wrapper" style={{ position: "relative" }}>
            <video ref={videoRef} width="640" autoPlay muted />

            {overlayColor && (
              <div
                style={{
                  position: "absolute",
                  top: 0,
                  left: 0,
                  width: "100%",
                  height: "100%",
                  backgroundColor: overlayColor,
                  opacity: 0.2,
                  zIndex: 1,
                }}
              />
            )}

            {overlayLabel && (
              <div
                style={{
                  position: "absolute",
                  top: "45%",
                  left: "50%",
                  transform: "translate(-50%, -50%)",
                  fontSize: "3rem",
                  fontWeight: "bold",
                  color: overlayColor,
                  backgroundColor: "rgba(255,255,255,0.85)",
                  padding: "10px 20px",
                  borderRadius: "10px",
                  zIndex: 2,
                }}
              >
                {overlayLabel}
              </div>
            )}

            {motionStatus && (
              <div
                style={{
                  position: "absolute",
                  top: "20%",
                  left: "50%",
                  transform: "translateX(-50%)",
                  fontSize: "2rem",
                  fontWeight: "bold",
                  backgroundColor: "rgba(0,0,0,0.5)",
                  color: "white",
                  padding: "8px 16px",
                  borderRadius: "10px",
                  zIndex: 2,
                }}
              >
                {motionStatus}
              </div>
            )}
          </div>
        </div>

        {/* 오른쪽: 상태 및 점수 */}
        <div className="right-panel">
          <div className="status-box">
            <h3>{status}</h3>

            {latestCycleScore && (
              <div className="cycle-feedback">
                최근 측정 점수: <strong>{latestCycleScore}점</strong> - {latestLabel}
              </div>
            )}

            {cycleScores.length > 0 && (
              <div className="score-list">
                <h4>측정 결과 ({cycleScores.length} / 3)</h4> {/* ✅ 회차 표시 */}
                {cycleScores.map((score, idx) => (
                  <div key={idx}>✅ Cycle {idx + 1}: {score.toFixed(1)}점</div>
                ))}
                {totalScore !== null && (
                  <div style={{ marginTop: "10px", fontWeight: "bold" }}>
                    총점: {totalScore.toFixed(2)} / 100
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
