import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './Hard.css';

export default function Hard() {
  const [name, setName] = useState('');
  const [score, setScore] = useState('');
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();

    const newEntry = {
        user_id: name,
        total_score: parseInt(score, 10)
    };

    try {
        await fetch("http://localhost:8000/api/hard-ranking", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(newEntry)
        });

        navigate("/hard-ranking");
    } catch (err) {
        alert("서버 저장 실패!");
        console.error(err);
    }
};


  return (
    <div className="page-container">
      <h1 className="page-title">🔴 HARD MODE</h1>
      <form onSubmit={handleSubmit} className="form-box">
        <label>이름</label>
        <input value={name} onChange={(e) => setName(e.target.value)} required />
        <label>점수</label>
        <input
          type="number"
          value={score}
          onChange={(e) => setScore(e.target.value)}
          required
        />
        <div className="button-row">
          <button type="submit" className="button">제출</button>
          <button type="button" onClick={() => navigate('/')} className="button">🏠 홈으로</button>
        </div>
      </form>
    </div>
  );
}
