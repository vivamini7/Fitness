import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './RankingPage.css';

export default function RankingPage() {
  const [ranking, setRanking] = useState([]);
  const navigate = useNavigate();

  const fetchRanking = () => {
    fetch("http://localhost:8000/ranking")
      .then(res => res.json())
      .then(data => setRanking(data.ranking || []))
      .catch(err => console.error("랭킹 불러오기 실패:", err));
  };

  const handleReset = async () => {
    if (!window.confirm("정말로 모든 랭킹을 초기화하시겠습니까?")) return;
    await fetch("http://localhost:8000/ranking/reset", {
      method: "POST"
    });
    fetchRanking();
  };

  useEffect(() => {
    fetchRanking();
  }, []);

  return (
    <div className="ranking-container">
      <h2 className="ranking-title">🏆 사용자 랭킹</h2>
      <div className="ranking-buttons">
        <button className="reset-btn" onClick={handleReset}>🔄 랭킹 초기화</button>
        <button className="home-btn" onClick={() => navigate("/")}>🏠 홈으로 돌아가기</button>
      </div>
      <table className="ranking-table">
        <thead>
          <tr>
            <th>순위</th>
            <th>이름</th>
            <th>점수</th>
          </tr>
        </thead>
        <tbody>
          {ranking.map((user, idx) => (
            <tr key={user.user_id}>
              <td>{idx + 1}</td>
              <td>{user.user_id}</td>
              <td>{user.total_score}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
