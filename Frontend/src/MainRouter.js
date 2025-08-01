import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import HomePage from './HomePage';
import App from './App';
import RankingPage from './RankingPage';
import Hard from './hard';
import HardRankingPage from './HardRankingPage'; // ✅ 새로 만들 파일

export default function MainRouter() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/app" element={<App />} />
        <Route path="/ranking" element={<RankingPage />} />
        <Route path="/hard" element={<Hard />} />
        <Route path="/hard-ranking" element={<HardRankingPage />} /> {/* ✅ 추가 */}
      </Routes>
    </Router>
  );
}
