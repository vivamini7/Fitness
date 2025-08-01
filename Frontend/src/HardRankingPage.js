import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './Hard.css';

export default function HardRankingPage() {
  const [rankings, setRankings] = useState([]);
  const navigate = useNavigate();

  useEffect(() => {
    fetch("http://localhost:8000/api/hard-ranking")
      .then(res => res.json())
      .then(data => {
        const sorted = [...data].sort((a, b) => b.total_score - a.total_score);
        setRankings(sorted);
      })
      .catch(err => {
        alert("랭킹 불러오기 실패!");
        console.error(err);
      });
  }, []);


  return (
    <div className="page-container">
      <h1 className="page-title">🏆 HARD MODE 랭킹</h1>
      <table className="ranking-table">
        <thead>
          <tr>
            <th>순위</th>
            <th>이름</th>
            <th>점수</th>
          </tr>
        </thead>
        <tbody>
          {rankings.map((entry, index) => (
            <tr key={index}>
              <td>{index + 1}위</td>
              <td>{entry.user_id}</td>
              <td>{entry.total_score}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="home-button">
        <button onClick={() => navigate('/')} className="button">🏠 홈으로</button>
      </div>
    </div>
  );
}
