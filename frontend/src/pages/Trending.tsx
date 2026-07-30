import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { 
  TrendingUp, 
  Flame, 
  RefreshCw, 
  Tag, 
  ExternalLink,
  Search,
  Filter
} from "lucide-react";

import apiClient from "../api/client";
import ArticleModal from "../components/ArticleModal";


interface Article {
  id: string;
  title: string;
  source: string;
  url: string;
  category: string;
  published_date: string;
  content?: string;
  cleaned_content?: string;
  summary?: string;
}

interface TrendingTopic {
  topic: string;
  count: number;
  category: string;
  velocity: string;
  description?: string;
  url?: string;
}


interface AnalyticsData {
  total_articles: number;
  sources_breakdown?: Record<string, number>;
  categories_breakdown?: Record<string, number>;
  trending_topics?: Array<{ topic: string; count: number; category: string }>;
}

export default function Trending() {
  const navigate = useNavigate();
  const [data, setData] = useState<AnalyticsData | null>(null);
  const [stats, setStats] = useState<any>(null);
  const [articles, setArticles] = useState<Article[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [selectedCategory, setSelectedCategory] = useState<string>("All");

  // Topic Modal State
  const [activeTopicModal, setActiveTopicModal] = useState<TrendingTopic | null>(null);
  const [modalArticles, setModalArticles] = useState<Article[]>([]);
  const [loadingModalArticles, setLoadingModalArticles] = useState<boolean>(false);


  const fetchData = async () => {
    setLoading(true);
    try {
      const [analyticsRes, articlesRes] = await Promise.all([
        apiClient.get("/api/analytics/metrics"),
        apiClient.get("/api/dashboard/stats")
      ]);
      setData(analyticsRes.data);
      setStats(articlesRes.data);
      setArticles(articlesRes.data.recent_articles || []);
    } catch (err) {
      console.error("Error fetching trending data:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
    window.addEventListener("liveNewsRefreshed", fetchData);
    return () => window.removeEventListener("liveNewsRefreshed", fetchData);
  }, []);

  const categoryCounts = data?.categories_breakdown || stats?.categories_breakdown || {};
  const totalArts = data?.total_articles || stats?.total_articles || 1;

  const backendTrendingTopics: any[] = (stats?.trending_topics || []).map((t: any) => ({
    topic: t.topic || "Headline Update",
    count: t.article_count || t.count || 1,
    category: t.category || "General News",
    velocity: t.velocity || "+15% growth",
    description: t.description || t.topic || "Real-time news cluster.",
    url: t.url,
    articles: t.articles || []
  }));

  const dynamicTopics: any[] = backendTrendingTopics.length > 0 
    ? backendTrendingTopics 
    : (Object.entries(categoryCounts) as [string, number][]).sort(([, a], [, b]) => b - a).map(([cat, count], idx) => {
        const topArt = articles.find(a => a.category === cat);
        return {
          topic: topArt?.title || `${cat} News Update`,
          count: count,
          category: cat,
          velocity: `+${Math.max(5, Math.round((Number(count) / totalArts) * 100) + (10 - idx * 2))}% growth`,
          description: (topArt?.cleaned_content || topArt?.content || topArt?.title || "").slice(0, 180) || `Real-time news stream in ${cat}.`,
          url: topArt?.url,
          articles: topArt ? [topArt] : []
        };
      });


  const categories = ["All", ...Object.keys(categoryCounts)];

  const filteredTopics = dynamicTopics.filter(item => {
    const matchesSearch = item.topic.toLowerCase().includes(searchQuery.toLowerCase()) || 
                          item.category.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesCategory = selectedCategory === "All" || item.category === selectedCategory;
    return matchesSearch && matchesCategory;
  });

  // Open Topic Modal and show exact cluster articles or fetch live articles
  const handleOpenTopicModal = async (topic: any) => {
    setActiveTopicModal(topic);
    setLoadingModalArticles(true);

    if (topic.articles && topic.articles.length > 0) {
      setModalArticles(topic.articles);
      setLoadingModalArticles(false);
      return;
    }

    try {
      // Stage 1: Try title-specific query search for articles on this exact story
      const titleRes = await apiClient.get("/api/dashboard/latest", {
        params: { query: topic.topic, limit: 15 }
      });
      if (titleRes.data && titleRes.data.articles && titleRes.data.articles.length > 0) {
        setModalArticles(titleRes.data.articles);
        setLoadingModalArticles(false);
        return;
      }

      // Stage 2: Try category lookup
      const catRes = await apiClient.get("/api/dashboard/by-category", {
        params: { category: topic.category, limit: 15 }
      });
      if (catRes.data && catRes.data.articles && catRes.data.articles.length > 0) {
        setModalArticles(catRes.data.articles);
        setLoadingModalArticles(false);
        return;
      }

      // Stage 3: In-memory category match or top 10 articles fallback
      const matched = articles.filter(a => (a.category || "").toLowerCase() === (topic.category || "").toLowerCase());
      setModalArticles(matched.length > 0 ? matched : articles.slice(0, 10));
    } catch (err) {
      console.error("Error fetching topic articles:", err);
      setModalArticles(articles.slice(0, 10));
    } finally {
      setLoadingModalArticles(false);
    }
  };






  return (
    <div className="space-y-8 max-w-7xl mx-auto pb-16">
      {/* Top Banner Header */}
      <div className="bg-gradient-to-r from-slate-900 via-indigo-950 to-slate-900 text-white p-6 md:p-8 rounded-3xl shadow-xl flex flex-col md:flex-row items-start md:items-center justify-between gap-6 border border-slate-800">
        <div className="space-y-2 max-w-2xl">
          <div className="flex items-center gap-2">
            <span className="p-2 bg-amber-500/20 rounded-xl text-amber-300 border border-amber-500/30">
              <Flame className="w-5 h-5 text-amber-400 animate-pulse" />
            </span>
            <span className="text-xs font-extrabold uppercase tracking-widest text-amber-400">
              Real-Time Clustering Intelligence
            </span>
          </div>
          <h1 className="text-3xl font-black text-white tracking-tight">
            Trending Topics & News Clusters
          </h1>
          <p className="text-slate-300 text-xs leading-relaxed">
            Click <strong>"Read Article"</strong> on any headline to open the exact news story page in a new browser tab.
          </p>
        </div>

        <button 
          onClick={fetchData} 
          disabled={loading}
          className="px-4 py-2.5 bg-indigo-600 hover:bg-indigo-700 text-white rounded-xl font-extrabold text-xs flex items-center gap-2 transition-all shadow-md cursor-pointer disabled:opacity-50 shrink-0"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? "animate-spin" : ""}`} />
          {loading ? "Refreshing..." : "Refresh Feed"}
        </button>
      </div>

      {/* Filter and Search Controls */}
      <div className="bg-white p-6 rounded-3xl border border-slate-200/90 shadow-xs space-y-4">
        <div className="flex flex-col md:flex-row items-stretch md:items-center justify-between gap-4">
          <div className="relative flex-1">
            <Search className="w-4 h-4 text-slate-400 absolute left-3.5 top-3.5" />
            <input 
              type="text"
              placeholder="Search trending topics, categories, or news keywords..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-4 py-2.5 bg-slate-50 border border-slate-200 rounded-2xl text-xs focus:outline-none focus:ring-2 focus:ring-amber-500/50 transition-all"
            />
          </div>

          <div className="flex items-center gap-2 overflow-x-auto pb-1 md:pb-0">
            <span className="text-xs font-bold text-slate-400 flex items-center gap-1 shrink-0 mr-1">
              <Filter className="w-3.5 h-3.5 text-amber-500" /> Category:
            </span>
            {categories.map(cat => (
              <button
                key={cat}
                onClick={() => setSelectedCategory(cat)}
                className={`px-3.5 py-1.5 rounded-xl text-xs font-bold transition-all cursor-pointer whitespace-nowrap ${
                  selectedCategory === cat
                    ? "bg-amber-500 text-white shadow-md shadow-amber-500/30"
                    : "bg-slate-100 text-slate-600 hover:bg-slate-200"
                }`}
              >
                {cat}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* TOPIC CLUSTERS GRID */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-black text-slate-900 flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-amber-500" />
            Top Clustered News Topics ({filteredTopics.length})
          </h2>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {filteredTopics.map((topic) => (
            <div 
              key={topic.topic}
              className="bg-white p-6 rounded-3xl border border-slate-200/90 shadow-xs hover:shadow-lg hover:border-amber-300 transition-all space-y-4 flex flex-col justify-between group"
            >
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="px-3 py-1 bg-amber-50 text-amber-700 font-black text-xs rounded-xl border border-amber-100 flex items-center gap-1.5 shadow-2xs">
                    <Tag className="w-3.5 h-3.5 text-amber-600" />
                    {topic.category}
                  </span>
                  <span className="text-[10px] font-extrabold text-emerald-700 bg-emerald-50 px-2.5 py-1 rounded-md border border-emerald-100">
                    {topic.velocity}
                  </span>
                </div>

                <h3 
                  onClick={() => handleOpenTopicModal(topic)}
                  className="text-base font-extrabold text-slate-900 leading-snug group-hover:text-amber-600 transition-colors cursor-pointer"
                >
                  {topic.topic}
                </h3>

                <p className="text-xs text-slate-500 line-clamp-2 leading-relaxed">
                  {topic.description}
                </p>

                <div className="flex items-center justify-between text-xs pt-2">
                  <span className="text-slate-500 font-medium">Topic Article Volume:</span>
                  <span className="font-extrabold text-amber-900 bg-amber-50 px-2.5 py-1 rounded-lg border border-amber-100">
                    {topic.count} Articles
                  </span>
                </div>
              </div>

              <div className="pt-3 border-t border-slate-100 flex items-center justify-between text-xs">
                <span className="text-slate-400 text-[11px] font-bold">News Cluster</span>
                <button
                  onClick={() => handleOpenTopicModal(topic)}
                  className="px-3.5 py-2 bg-amber-600 hover:bg-amber-700 text-white text-xs font-extrabold rounded-xl border border-amber-500 flex items-center gap-1.5 transition-all cursor-pointer shadow-md hover:shadow-lg active:scale-95"
                >
                  View Topic Articles <ExternalLink className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* REUSABLE ARTICLE CLUSTER MODAL */}
      <ArticleModal
        isOpen={Boolean(activeTopicModal)}
        onClose={() => setActiveTopicModal(null)}
        title={activeTopicModal?.topic || ""}
        category={activeTopicModal?.category}
        badgeText={activeTopicModal?.velocity}
        description={activeTopicModal?.description}
        primaryUrl={activeTopicModal?.url}
        articles={modalArticles}
        loading={loadingModalArticles}
        accentColor="amber"
        onAskAI={(topicName) => {
          navigate("/chat", { state: { query: `Tell me about the trending news topic: ${topicName}` } });
        }}
      />
    </div>
  );
}


