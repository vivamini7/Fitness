// ✅ MainRouter.js
import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import HomePage from './HomePage';
import App from './App';
import RankingPage from './RankingPage';

export default function MainRouter() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/app" element={<App />} />
        <Route path="/ranking" element={<RankingPage />} />
      </Routes>
    </Router>
  );
}
