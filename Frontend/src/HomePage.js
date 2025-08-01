// src/HomePage.js
import React from 'react';
import { useNavigate } from 'react-router-dom';
import './HomePage.css';

export default function HomePage() {
  const navigate = useNavigate();

  const selectMode = (mode) => {
    if (mode === "hard") {
      navigate("/hard");
    } else {
      navigate(`/app?mode=${mode}`);
    }
  };


  return (
    <div className="home-container">
      <div className="half left" onClick={() => selectMode("easy")}>
        <h1>🟢 EASY MODE</h1>
      </div>
      <div className="half right" onClick={() => selectMode("hard")}>
        <h1>🔴 HARD MODE</h1>
      </div>
    </div>
  );
}
