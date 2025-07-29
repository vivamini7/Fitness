// ✅ index.js
import React from 'react';
import ReactDOM from 'react-dom/client';
import MainRouter from './MainRouter';
import './index.css';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <MainRouter />
  </React.StrictMode>
);
