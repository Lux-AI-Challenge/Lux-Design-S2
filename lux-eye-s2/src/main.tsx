import { MantineProvider } from '@mantine/core';
import { ModalsProvider } from '@mantine/modals';
import { NotificationsProvider } from '@mantine/notifications';
import React from 'react';
import ReactDOM from 'react-dom/client';
import { HashRouter, Route, Routes } from 'react-router-dom';
import { App } from './App';
import { HomePage } from './pages/home/HomePage';
import { LeaderboardPage } from './pages/leaderboard/LeaderboardPage';
import { NotebookPage } from './pages/notebook/NotebookPage';
import { OpenPage } from './pages/open/OpenPage';
import { VisualizerPage } from './pages/visualizer/VisualizerPage';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <MantineProvider withGlobalStyles withNormalizeCSS>
      <NotificationsProvider position="top-center">
        <ModalsProvider>
          <HashRouter>
            <Routes>
              <Route path="/" element={<App />}>
                <Route path="/" element={<HomePage />} />
                <Route path="/visualizer" element={<VisualizerPage />} />
              </Route>
              {/* /kaggle for backwards-compatibility */}
              <Route path="/kaggle" element={<NotebookPage />} />
              <Route path="/notebook" element={<NotebookPage />} />
              <Route path="/leaderboard" element={<LeaderboardPage />} />
              <Route path="/open" element={<OpenPage />} />
            </Routes>
          </HashRouter>
        </ModalsProvider>
      </NotificationsProvider>
    </MantineProvider>
  </React.StrictMode>,
);
