import { BrowserRouter, Routes, Route } from 'react-router-dom';
import MainLayout from './layouts/MainLayout';
import Dashboard from './pages/Dashboard';
import Trending from './pages/Trending';
import StateTrending from './pages/StateTrending';
import Chat from './pages/Chat';
import Compare from './pages/Compare';
import Analytics from './pages/Analytics';
import Evaluation from './pages/Evaluation';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<MainLayout />}>
          <Route index element={<Dashboard />} />
          <Route path="trending" element={<Trending />} />
          <Route path="state-trending" element={<StateTrending />} />
          <Route path="chat" element={<Chat />} />
          <Route path="compare" element={<Compare />} />
          <Route path="analytics" element={<Analytics />} />
          <Route path="evaluation" element={<Evaluation />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;
