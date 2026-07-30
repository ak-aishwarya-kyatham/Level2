import { useState } from "react";
import { Outlet, Link, useLocation } from "react-router-dom";
import { 
  LayoutDashboard, 
  MessageSquare, 
  GitCompare, 
  BarChart3, 
  RefreshCw, 
  Radio, 
  CheckCircle2, 
  Sparkles,
  Newspaper,
  ShieldCheck,
  Flame,
  MapPin
} from "lucide-react";
import apiClient from "../api/client";


export default function MainLayout() {
  const location = useLocation();
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [lastRefreshed, setLastRefreshed] = useState<string>("Just now");
  const [toastMessage, setToastMessage] = useState<string | null>(null);

  const handleRefresh = async () => {
    setIsRefreshing(true);
    try {
      const res = await apiClient.post("/api/dashboard/refresh");
      setLastRefreshed(new Date().toLocaleTimeString());
      setToastMessage(`Fetched live news! Total articles: ${res.data.total_articles}`);
      
      // Dispatch global refresh event for active page components to reload
      window.dispatchEvent(new CustomEvent("liveNewsRefreshed"));
      
      setTimeout(() => setToastMessage(null), 4000);
    } catch (err) {
      console.error("Refresh error:", err);
      setToastMessage("Failed to fetch live news feeds.");
      setTimeout(() => setToastMessage(null), 4000);
    } finally {
      setIsRefreshing(false);
    }
  };

  const navItems = [
    { path: "/", label: "Live Dashboard", icon: LayoutDashboard },
    { path: "/trending", label: "Trending Topics", icon: Flame },
    { path: "/state-trending", label: "State-Wise Trending", icon: MapPin },
    { path: "/chat", label: "AI Chat Assistant", icon: MessageSquare },
    { path: "/compare", label: "Compare News", icon: GitCompare },
    { path: "/analytics", label: "Analytics & Trends", icon: BarChart3 },
    { path: "/evaluation", label: "Evaluation Metrics", icon: ShieldCheck },
  ];

  return (
    <div className="flex h-screen bg-slate-50 font-sans antialiased text-slate-900">
      {/* Sidebar */}
      <aside className="w-64 bg-slate-900 text-white flex flex-col border-r border-slate-800 shadow-xl z-20">
        <div className="p-6 border-b border-slate-800/80">
          <div className="flex items-center gap-3">
            <div className="p-2.5 bg-gradient-to-tr from-indigo-600 to-violet-500 rounded-xl shadow-lg shadow-indigo-500/30">
              <Newspaper className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-tight bg-gradient-to-r from-white via-slate-100 to-indigo-200 bg-clip-text text-transparent">
                NewsIntel AI
              </h1>
              <span className="text-xs font-semibold text-indigo-400 uppercase tracking-widest block">
                Multi-Agent Platform
              </span>
            </div>
          </div>
        </div>

        {/* Live Status Card in Sidebar */}
        <div className="mx-4 my-4 p-3.5 bg-slate-800/60 rounded-xl border border-slate-700/50 backdrop-blur-sm">
          <div className="flex items-center justify-between text-xs mb-1.5">
            <span className="text-slate-400 font-medium flex items-center gap-1.5">
              <Radio className="w-3.5 h-3.5 text-emerald-400 animate-pulse-slow" />
              Live Ingestion
            </span>
            <span className="text-emerald-400 font-semibold px-2 py-0.5 bg-emerald-500/10 rounded-full text-[10px]">
              ONLINE
            </span>
          </div>
          <p className="text-[11px] text-slate-400">
            Continuously polling top RSS feeds & Google News
          </p>
        </div>

        <nav className="flex-1 px-3 py-2 space-y-1.5 overflow-y-auto">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = location.pathname === item.path;
            return (
              <Link
                key={item.path}
                to={item.path}
                className={`flex items-center gap-3 px-3.5 py-3 rounded-xl text-sm font-medium transition-all duration-200 ${
                  isActive
                    ? "bg-gradient-to-r from-indigo-600 to-violet-600 text-white shadow-lg shadow-indigo-600/30 font-semibold"
                    : "text-slate-400 hover:bg-slate-800/80 hover:text-white"
                }`}
              >
                <Icon className={`w-5 h-5 ${isActive ? "text-white" : "text-slate-400"}`} />
                {item.label}
              </Link>
            );
          })}
        </nav>

        {/* Footer info */}
        <div className="p-4 border-t border-slate-800 text-xs text-slate-500 flex items-center justify-between">
          <span>Engineered with LangGraph</span>
          <Sparkles className="w-4 h-4 text-indigo-400" />
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col overflow-hidden">
        {/* Top Header */}
        <header className="bg-white/95 backdrop-blur-md border-b border-slate-200/80 h-16 flex items-center px-6 justify-between shadow-xs z-10">
          <div className="flex items-center gap-3">
            <h2 className="text-lg font-bold text-slate-800">
              {navItems.find((i) => i.path === location.pathname)?.label || "Overview"}
            </h2>
            <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-indigo-50 text-indigo-700 border border-indigo-100 flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-indigo-600 animate-ping"></span>
              Live News Stream
            </span>
          </div>

          <div className="flex items-center gap-4">
            <div className="text-xs text-slate-500 text-right hidden sm:block">
              <span className="block font-medium">Auto-Sync Active</span>
              <span className="text-slate-400">Last sync: {lastRefreshed}</span>
            </div>

            <button
              onClick={handleRefresh}
              disabled={isRefreshing}
              className="flex items-center gap-2 px-4 py-2 bg-indigo-600 hover:bg-indigo-700 disabled:bg-indigo-400 text-white text-xs font-semibold rounded-xl shadow-sm transition-all duration-200 hover:shadow-md cursor-pointer"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${isRefreshing ? "animate-spin" : ""}`} />
              {isRefreshing ? "Fetching Live Feeds..." : "Refresh Feeds"}
            </button>
          </div>
        </header>

        {/* Toaster Notification */}
        {toastMessage && (
          <div className="bg-emerald-600 text-white text-xs font-semibold px-4 py-2.5 flex items-center justify-between shadow-lg animate-in slide-in-from-top duration-300">
            <div className="flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4" />
              <span>{toastMessage}</span>
            </div>
            <button 
              onClick={() => setToastMessage(null)} 
              className="text-emerald-100 hover:text-white text-sm font-bold ml-4 cursor-pointer"
            >
              ✕
            </button>
          </div>
        )}

        {/* Page Content */}
        <div className="flex-1 overflow-auto p-6 bg-slate-50/50">
          <Outlet />
        </div>
      </main>
    </div>
  );
}
