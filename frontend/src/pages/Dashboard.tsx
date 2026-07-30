import { useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";

import { 
  Newspaper, 
  Layers, 
  Globe2, 
  ShieldCheck, 
  TrendingUp, 
  Search, 
  Filter, 
  ExternalLink, 
  MessageSquareText, 
  Clock, 
  X, 
  BookOpen,
  Sparkles,
  RefreshCw,
  AlertCircle
} from "lucide-react";
import apiClient from "../api/client";
import MetricCard from "../components/MetricCard";


interface Article {
  id: string;
  title: string;
  content: string;
  cleaned_content?: string;
  source: string;
  url: string;
  category: string;
  published_date: string;
  created_at: string;
}

interface TrendingTopic {
  rank: number;
  topic: string;
  article_count: number;
  url: string;
}

interface Stats {
  total_articles: number;
  categories_count: number;
  sources_count: number;
  duplicates_avoided: number;
  trending_topics: TrendingTopic[];
}

export default function Dashboard() {
  const navigate = useNavigate();
  const isInitialMount = useRef(true);
  const [stats, setStats] = useState<Stats>({
    total_articles: 0,
    categories_count: 0,
    sources_count: 0,
    duplicates_avoided: 0,
    trending_topics: []
  });

  const [articles, setArticles] = useState<Article[]>([]);
  const [sources, setSources] = useState<string[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [searching, setSearching] = useState<boolean>(false);
  const [fetchError, setFetchError] = useState<string | null>(null);

  // Filters
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [selectedCategory, setSelectedCategory] = useState<string>("All");
  const [selectedSource, setSelectedSource] = useState<string>("All");

  // Selected article for detailed modal
  const [selectedArticle, setSelectedArticle] = useState<Article | null>(null);

  const categories = [
    "All",
    "Technology",
    "Business",
    "Politics",
    "Sports",
    "Health",
    "International",
    "General News"
  ];

  const fetchStats = async () => {
    try {
      const res = await apiClient.get("/api/dashboard/stats");
      setStats(res.data);
    } catch (err) {
      console.error("Error fetching stats:", err);
    }
  };

  const fetchSources = async () => {
    try {
      const res = await apiClient.get("/api/sources/");
      setSources(res.data.sources || []);
    } catch (err) {
      console.error("Error fetching sources:", err);
    }
  };

  const fetchArticles = async (query = searchQuery, cat = selectedCategory, src = selectedSource) => {
    setSearching(true);
    setFetchError(null);
    try {
      const res = await apiClient.get("/api/dashboard/latest", {
        params: {
          query: query,
          category: cat,
          source: src,
          limit: 50
        }
      });
      if (res.data && res.data.error) {
        setFetchError(res.data.error);
        setArticles([]);
      } else {
        setArticles(res.data.articles || []);
      }
    } catch (err) {
      console.error("Error fetching articles:", err);
      setFetchError("Failed to fetch articles from backend API. Please check server connection.");
    } finally {
      setSearching(false);
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchStats();
    fetchSources();
    fetchArticles();

    // Listen to global live news refresh event from header
    const handleGlobalRefresh = () => {
      fetchStats();
      fetchSources();
      fetchArticles();
    };

    window.addEventListener("liveNewsRefreshed", handleGlobalRefresh);
    return () => window.removeEventListener("liveNewsRefreshed", handleGlobalRefresh);
  }, []);

  // Debounced search trigger (skip initial mount to prevent race conditions)
  useEffect(() => {
    if (isInitialMount.current) {
      isInitialMount.current = false;
      return;
    }
    const handler = setTimeout(() => {
      fetchArticles(searchQuery, selectedCategory, selectedSource);
    }, 400);
    return () => clearTimeout(handler);
  }, [searchQuery, selectedCategory, selectedSource]);

  const handleAskAI = (articleTitle: string) => {
    navigate("/chat", { state: { prefill: `Tell me more about "${articleTitle}" and summarize its implications.` } });
  };

  const formatDate = (dateStr: string) => {
    if (!dateStr) return "Recent";
    try {
      const d = new Date(dateStr);
      if (isNaN(d.getTime())) return "Recently published";
      return d.toLocaleDateString(undefined, {
        month: "short",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit"
      });
    } catch {
      return "Recent";
    }
  };

  return (
    <div className="space-y-8 max-w-7xl mx-auto">
      {/* Header Banner */}
      <div className="bg-gradient-to-r from-indigo-900 via-slate-900 to-indigo-950 rounded-2xl p-6 md:p-8 text-white shadow-xl relative overflow-hidden">
        <div className="absolute right-0 top-0 w-96 h-96 bg-indigo-500/10 rounded-full blur-3xl pointer-events-none"></div>
        <div className="relative z-10 flex flex-col md:flex-row md:items-center justify-between gap-6">
          <div>
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-500/20 text-indigo-300 border border-indigo-500/30 text-xs font-semibold mb-3">
              <Sparkles className="w-3.5 h-3.5" />
              Live News Intelligence & Analytics
            </div>
            <h1 className="text-2xl md:text-3xl font-extrabold tracking-tight">
              Real-Time Media Aggregation Engine
            </h1>
            <p className="text-slate-300 text-sm mt-2 max-w-2xl leading-relaxed">
              Monitoring, categorizing, deduplicating, and synthesizing live feeds from major news outlets and Google News in real-time.
            </p>
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={() => {
                fetchStats();
                fetchArticles();
              }}
              className="px-4 py-2.5 bg-indigo-600 hover:bg-indigo-500 text-white rounded-xl text-xs font-semibold shadow-md transition-all flex items-center gap-2 cursor-pointer"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${searching ? "animate-spin" : ""}`} />
              Sync Live Articles
            </button>
          </div>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
        <MetricCard
          title="Total Live Articles"
          value={stats.total_articles.toLocaleString()}
          subtitle="Real-time RSS index"
          icon={Newspaper}
          iconBgColor="bg-indigo-50"
          iconTextColor="text-indigo-600"
          badgeText="Verified Stream"
          badgeColor="indigo"
        />

        <MetricCard
          title="Active Categories"
          value={stats.categories_count}
          subtitle="Categorized domains"
          icon={Layers}
          iconBgColor="bg-emerald-50"
          iconTextColor="text-emerald-600"
          badgeText="Structured"
          badgeColor="emerald"
        />

        <MetricCard
          title="Indexed Outlets"
          value={stats.sources_count}
          subtitle="Top publisher feeds"
          icon={Globe2}
          iconBgColor="bg-blue-50"
          iconTextColor="text-blue-600"
          badgeText="Multi-Source"
          badgeColor="indigo"
        />

        <MetricCard
          title="Duplicates Filtered"
          value={stats.duplicates_avoided}
          subtitle="Redundancies avoided"
          icon={ShieldCheck}
          iconBgColor="bg-amber-50"
          iconTextColor="text-amber-600"
          badgeText="Cleaned"
          badgeColor="amber"
        />
      </div>


      {/* Main Grid: Live Feed + Trending Topics */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Left Column: Filters & Articles (2 cols) */}
        <div className="lg:col-span-2 space-y-6">
          {/* Search & Filter Controls */}
          <div className="bg-white p-5 rounded-2xl shadow-xs border border-slate-200/80 space-y-4">
            {/* Search Input */}
            <div className="relative">
              <Search className="w-5 h-5 absolute left-3.5 top-3.5 text-slate-400" />
              <input
                type="text"
                placeholder="Search live news keywords or topics (e.g., AI, Economy, Sports, Tech)..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-11 pr-10 py-3 bg-slate-50 border border-slate-200 rounded-xl text-sm focus:bg-white focus:outline-none focus:ring-2 focus:ring-indigo-500 transition-all font-medium"
              />
              {searchQuery && (
                <button
                  onClick={() => setSearchQuery("")}
                  className="absolute right-3.5 top-3.5 text-slate-400 hover:text-slate-600 cursor-pointer"
                >
                  <X className="w-4 h-4" />
                </button>
              )}
            </div>

            {/* Category Pills */}
            <div className="flex items-center gap-2 overflow-x-auto pb-1 scrollbar-none">
              <Filter className="w-4 h-4 text-slate-400 shrink-0 mr-1" />
              {categories.map((cat) => (
                <button
                  key={cat}
                  onClick={() => setSelectedCategory(cat)}
                  className={`px-3.5 py-1.5 rounded-xl text-xs font-semibold whitespace-nowrap transition-all cursor-pointer ${
                    selectedCategory === cat
                      ? "bg-indigo-600 text-white shadow-sm"
                      : "bg-slate-100 text-slate-600 hover:bg-slate-200"
                  }`}
                >
                  {cat}
                </button>
              ))}
            </div>

            {/* Source Selector */}
            <div className="flex items-center justify-between pt-2 border-t border-slate-100 text-xs text-slate-500">
              <span>Filtering by news source:</span>
              <select
                value={selectedSource}
                onChange={(e) => setSelectedSource(e.target.value)}
                className="bg-slate-50 border border-slate-200 rounded-lg px-3 py-1.5 text-slate-700 font-medium focus:outline-none focus:ring-2 focus:ring-indigo-500 cursor-pointer"
              >
                <option value="All">All Outlets</option>
                {sources.map((src) => (
                  <option key={src} value={src}>
                    {src}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Articles Feed */}
          <div className="space-y-4">
            <div className="flex items-center justify-between px-1">
              <h2 className="text-base font-bold text-slate-800 flex items-center gap-2">
                <span>Live Feed</span>
                <span className="text-xs px-2 py-0.5 bg-slate-200 text-slate-700 rounded-full font-medium">
                  {articles.length} items
                </span>
              </h2>
              {searching && (
                <span className="text-xs font-semibold text-indigo-600 flex items-center gap-1.5 animate-pulse">
                  <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                  Updating stream...
                </span>
              )}
            </div>

            {loading ? (
              <div className="space-y-4">
                {[1, 2, 3].map((n) => (
                  <div key={n} className="bg-white p-5 rounded-2xl shadow-xs border border-slate-200/80 animate-pulse space-y-3">
                    <div className="h-4 bg-slate-200 rounded w-3/4"></div>
                    <div className="h-3 bg-slate-100 rounded w-full"></div>
                    <div className="h-3 bg-slate-100 rounded w-2/3"></div>
                  </div>
                ))}
              </div>
            ) : fetchError ? (
              <div className="bg-rose-50 p-6 rounded-2xl text-center border border-rose-200 space-y-3">
                <AlertCircle className="w-10 h-10 text-rose-500 mx-auto" />
                <h3 className="text-sm font-bold text-rose-800">Error Connecting to Backend</h3>
                <p className="text-xs text-rose-600 max-w-sm mx-auto">{fetchError}</p>
                <button
                  onClick={() => fetchArticles(searchQuery, selectedCategory, selectedSource)}
                  className="px-4 py-2 bg-rose-600 text-white rounded-xl text-xs font-semibold hover:bg-rose-700 cursor-pointer shadow-xs transition-colors"
                >
                  Retry Fetching Articles
                </button>
              </div>
            ) : articles.length === 0 ? (
              <div className="bg-white p-8 rounded-2xl text-center border border-slate-200/80 space-y-3">
                <Newspaper className="w-10 h-10 text-slate-300 mx-auto" />
                <h3 className="text-sm font-bold text-slate-700">No articles matching current filters</h3>
                <p className="text-xs text-slate-500 max-w-sm mx-auto">
                  Try clearing search terms or clicking "Sync Live Articles" above or "Refresh Feeds" in the top bar to fetch fresh RSS stories.
                </p>
                <button
                  onClick={() => {
                    setSearchQuery("");
                    setSelectedCategory("All");
                    setSelectedSource("All");
                  }}
                  className="px-4 py-2 bg-indigo-50 text-indigo-600 rounded-xl text-xs font-semibold hover:bg-indigo-100 cursor-pointer"
                >
                  Reset All Filters
                </button>
              </div>
            ) : (
              <div className="space-y-4">
                {articles.map((art) => (
                  <div
                    key={art.id}
                    className="bg-white p-5 rounded-2xl shadow-xs border border-slate-200/80 hover:shadow-md hover:border-slate-300 transition-all space-y-3 group"
                  >
                    <div className="flex items-start justify-between gap-4">
                      <div className="flex items-center gap-2 flex-wrap text-xs">
                        <span className="px-2.5 py-1 bg-indigo-50 text-indigo-700 font-bold rounded-lg border border-indigo-100/80">
                          {art.source}
                        </span>
                        <span className="px-2 py-0.5 bg-slate-100 text-slate-600 font-medium rounded-md text-[11px]">
                          {art.category}
                        </span>
                        <span className="text-slate-400 text-[11px] flex items-center gap-1">
                          <Clock className="w-3 h-3" />
                          {formatDate(art.published_date)}
                        </span>
                      </div>

                      <a
                        href={art.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-slate-400 hover:text-indigo-600 transition-colors p-1"
                        title="View Original Source"
                      >
                        <ExternalLink className="w-4 h-4" />
                      </a>
                    </div>

                    <h3
                      onClick={() => setSelectedArticle(art)}
                      className="text-base font-bold text-slate-900 hover:text-indigo-600 transition-colors cursor-pointer leading-snug"
                    >
                      {art.title}
                    </h3>

                    <p className="text-xs text-slate-600 line-clamp-2 leading-relaxed">
                      {art.cleaned_content || art.content}
                    </p>

                    <div className="flex items-center justify-between pt-2 border-t border-slate-100 text-xs">
                      <button
                        onClick={() => setSelectedArticle(art)}
                        className="text-indigo-600 hover:text-indigo-700 font-semibold flex items-center gap-1 cursor-pointer"
                      >
                        <BookOpen className="w-3.5 h-3.5" />
                        Read Article Summary
                      </button>

                      <button
                        onClick={() => handleAskAI(art.title)}
                        className="px-3 py-1.5 bg-slate-100 hover:bg-indigo-50 hover:text-indigo-700 text-slate-700 font-medium rounded-lg transition-colors flex items-center gap-1.5 text-[11px] cursor-pointer"
                      >
                        <MessageSquareText className="w-3.5 h-3.5 text-indigo-600" />
                        Ask AI Assistant
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Right Column: Trending Topics & Intelligence Panel */}
        <div className="space-y-6">
          {/* Top 5 Dynamic Trending Topics */}
          <div className="bg-white p-6 rounded-2xl shadow-xs border border-slate-200/80 space-y-4">
            <div className="flex items-center justify-between border-b border-slate-100 pb-3">
              <h2 className="text-base font-bold text-slate-900 flex items-center gap-2">
                <TrendingUp className="w-5 h-5 text-indigo-600" />
                Live Trending Topics
              </h2>
              <span className="text-[11px] font-semibold text-slate-400 uppercase">Top Rank</span>
            </div>

            <div className="space-y-3">
              {stats.trending_topics.length === 0 ? (
                <p className="text-xs text-slate-400">Syncing live trends...</p>
              ) : (
                stats.trending_topics.map((item) => (
                  <a
                    key={item.rank}
                    href={item.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-start justify-between gap-3 p-3 rounded-xl hover:bg-slate-50 border border-transparent hover:border-slate-200 transition-all cursor-pointer group"
                  >
                    <div className="flex items-start gap-3">
                      <span className="w-6 h-6 rounded-lg bg-indigo-50 text-indigo-700 font-bold text-xs flex items-center justify-center shrink-0 group-hover:bg-indigo-600 group-hover:text-white transition-colors">
                        {item.rank}
                      </span>
                      <div>
                        <h4 className="text-xs font-bold text-slate-800 group-hover:text-indigo-600 transition-colors line-clamp-2 leading-snug">
                          {item.topic}
                        </h4>
                        <span className="text-[10px] text-slate-400 mt-1 block">
                          {item.article_count} related live feed{item.article_count > 1 ? "s" : ""}
                        </span>
                      </div>
                    </div>
                    <ExternalLink className="w-3.5 h-3.5 text-slate-300 group-hover:text-indigo-500 shrink-0 mt-0.5" />
                  </a>
                ))
              )}
            </div>
          </div>

          {/* AI Multi-Agent Workflow Info */}
          <div className="bg-gradient-to-br from-slate-900 to-indigo-950 p-6 rounded-2xl text-white shadow-lg space-y-4">
            <h3 className="text-sm font-bold text-indigo-300 flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-indigo-400" />
              Agentic Processing Pipeline
            </h3>
            <ul className="space-y-2.5 text-xs text-slate-300">
              <li className="flex items-start gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 mt-1.5 shrink-0"></span>
                <span><strong>Ingestion Agent:</strong> Real-time RSS & Google News scrapers.</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-blue-400 mt-1.5 shrink-0"></span>
                <span><strong>Cleaning & Categorizer:</strong> Strips noise & tags categories.</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-violet-400 mt-1.5 shrink-0"></span>
                <span><strong>Deduplication Agent:</strong> Prevents redundant media reporting.</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-amber-400 mt-1.5 shrink-0"></span>
                <span><strong>LangGraph Router:</strong> Grounded responses with citations.</span>
              </li>
            </ul>
          </div>
        </div>
      </div>

      {/* Article Detail Drawer / Modal */}
      {selectedArticle && (
        <div className="fixed inset-0 z-50 bg-slate-900/60 backdrop-blur-xs flex items-center justify-center p-4">
          <div className="bg-white rounded-2xl max-w-2xl w-full max-h-[85vh] flex flex-col shadow-2xl border border-slate-200 animate-in zoom-in-95 duration-200">
            {/* Modal Header */}
            <div className="p-6 border-b border-slate-100 flex items-start justify-between gap-4">
              <div>
                <div className="flex items-center gap-2 mb-2">
                  <span className="px-2.5 py-0.5 bg-indigo-50 text-indigo-700 text-xs font-bold rounded-md">
                    {selectedArticle.source}
                  </span>
                  <span className="px-2 py-0.5 bg-slate-100 text-slate-600 text-xs font-medium rounded-md">
                    {selectedArticle.category}
                  </span>
                </div>
                <h2 className="text-lg font-bold text-slate-900 leading-snug">
                  {selectedArticle.title}
                </h2>
              </div>
              <button
                onClick={() => setSelectedArticle(null)}
                className="p-1 text-slate-400 hover:text-slate-600 rounded-lg hover:bg-slate-100 cursor-pointer"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Modal Content */}
            <div className="p-6 overflow-y-auto space-y-4 text-sm text-slate-700 leading-relaxed">
              <div className="bg-slate-50 p-4 rounded-xl text-xs text-slate-500 flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                <span>Published: {formatDate(selectedArticle.published_date)}</span>
                <a
                  href={selectedArticle.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-[11px] text-indigo-600 hover:underline font-mono flex items-center gap-1 truncate max-w-md"
                >
                  <ExternalLink className="w-3 h-3 shrink-0" />
                  {selectedArticle.url}
                </a>
              </div>


              <div>
                <h4 className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-2">Full Cleaned Text</h4>
                <p className="whitespace-pre-wrap leading-relaxed text-slate-800">
                  {selectedArticle.cleaned_content || selectedArticle.content}
                </p>
              </div>
            </div>

            {/* Modal Footer */}
            <div className="p-4 border-t border-slate-100 bg-slate-50/50 rounded-b-2xl flex items-center justify-between">
              <a
                href={selectedArticle.url}
                target="_blank"
                rel="noopener noreferrer"
                className="px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded-xl text-xs font-semibold flex items-center gap-2 shadow-sm cursor-pointer"
              >
                <span>Read Full Article on Web</span>
                <ExternalLink className="w-3.5 h-3.5" />
              </a>

              <button
                onClick={() => {
                  const title = selectedArticle.title;
                  setSelectedArticle(null);
                  handleAskAI(title);
                }}
                className="px-4 py-2 bg-slate-900 hover:bg-slate-800 text-white rounded-xl text-xs font-semibold flex items-center gap-2 cursor-pointer"
              >
                <MessageSquareText className="w-3.5 h-3.5 text-indigo-400" />
                Ask AI Assistant
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
