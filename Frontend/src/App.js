import React, { useEffect, useRef, useState, useCallback } from 'react';
import './App.css';

export default function App() {
  const videoRef = useRef(null);
  const [status, setStatus] = useState("대기 중...");
  const [angles, setAngles] = useState([]);
  const [rankings, setRankings] = useState([]);
  const [userId, setUserId] = useState("");
  const [started, setStarted] = useState(false);
  const [scoreDetail, setScoreDetail] = useState(null);
  const [totalScore, setTotalScore] = useState(null);
  const [isSending, setIsSending] = useState(false);

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

      const blob = await new Promise((resolve) =>
        canvas.toBlob(resolve, 'image/jpeg', 0.8)
      );
      const formData = new FormData();
      formData.append('image', blob, 'frame.jpg');
      formData.append('user_id', userId);

      const response = await fetch("http://localhost:8000/analyze", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      setStatus(data.status);
      setAngles(data.angles || []);
      setScoreDetail(data.score_detail || null);
      setTotalScore(data.total_score || null);

      // ...existing code...
      if ((data.angles || []).length === 5) {
        const rankRes = await fetch("http://localhost:8000/ranking");
        const rankData = await rankRes.json();
        setRankings(rankData.ranking);

        // 마지막 점수 갱신을 위해 한 번 더 analyze 호출
        const lastRes = await fetch("http://localhost:8000/analyze", {
          method: "POST",
          body: formData,
        });
        const lastData = await lastRes.json();
        setScoreDetail(lastData.score_detail || null);
        setTotalScore(lastData.total_score || null);

        setStarted(false);
        setStatus("✅ 측정 완료! 다시 시작하려면 이름을 바꾸세요.");
      }
    } catch (err) {
      console.error("❌ Fetch error:", err);
    } finally {
      setIsSending(false);
    }
  }, [userId, isSending]);

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
    setAngles([]);
    setScoreDetail(null);
    setTotalScore(null);
    setStarted(true);
    setStatus("📸 측정 시작!");
  };

  return (
    <div className="container">
      <h1 className="title">🏋️‍♂️ 스쿼트 자세 분석기</h1>

      <div className="main-content">
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

          <div className="video-wrapper">
            <video ref={videoRef} width="640" autoPlay muted />
            <div className="status-box">
              <h3>{status}</h3>
              {angles.length > 0 && (
                <div className="angles">
                  {angles.map((a, idx) => (
                    <div key={idx} className="angle-item">
                      <strong>{idx + 1} cycle</strong>
                      <div>Trunk: {a.trunk_angle?.toFixed(2) ?? 'N/A'}°</div>
                      <div>Hip: {a.hip_angle?.toFixed(2) ?? 'N/A'}°</div>
                      <div>Knee: {a.knee_angle?.toFixed(2) ?? 'N/A'}°</div>
                      <div>Valgus: {a.knee_valgus_dist?.toFixed(2) ?? 'N/A'}px</div>
                    </div>
                  ))}
                </div>
              )}

              {scoreDetail && (
                <div className="score-box">
                  <h4>📊 점수 상세</h4>
                  <ul>
                    <li>Trunk: {scoreDetail.trunk_angle?.toFixed(2) ?? 'N/A'}점</li>
                    <li>Hip: {scoreDetail.hip_angle?.toFixed(2) ?? 'N/A'}점</li>
                    <li>Knee: {scoreDetail.knee_angle?.toFixed(2) ?? 'N/A'}점</li>
                    <li>Valgus: {scoreDetail.knee_valgus_dist?.toFixed(2) ?? 'N/A'}점</li>
                  </ul>
                  <h4 className="total-score">총점: {totalScore ?? 'N/A'} / 100</h4>
                </div>
              )}
            </div>
          </div>
        </div>

        {rankings.length > 0 && (
          <div className="ranking-box">
            <h3>🏆 전체 사용자 랭킹</h3>
            <table>
              <thead>
                <tr>
                  <th>순위</th>
                  <th>사용자</th>
                  <th>총점</th>
                </tr>
              </thead>
              <tbody>
                {rankings.map((r, idx) => (
                  <tr key={idx}>
                    <td>#{idx + 1}</td>
                    <td>{r.user_id}</td>
                    <td>{r.total_score?.toFixed(2) ?? '?'}점</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
