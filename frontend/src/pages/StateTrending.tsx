import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { 
  MapPin, 
  Search, 
  ExternalLink, 
  Building2, 
  Compass,
  ArrowRight,
  RefreshCw
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
  state?: string;
  summary?: string;
}

interface StateTrendingData {
  state: string;
  code: string;
  trendingTopic: string;
  category: string;
  articleCount: number;
  velocity: string;
  description: string;
  articles: Article[];
}

export default function StateTrending() {
  const navigate = useNavigate();
  const [stateSearchQuery, setStateSearchQuery] = useState<string>("");
  const [activeSearch, setActiveSearch] = useState<string>("");
  const [selectedState, setSelectedState] = useState<string>("All");
  
  // Live Articles Cache per State
  const [liveStateArticles, setLiveStateArticles] = useState<Record<string, Article[]>>({});
  const [loadingStateNews, setLoadingStateNews] = useState<boolean>(false);

  // Modal state for state article list
  const [activeStateModal, setActiveStateModal] = useState<StateTrendingData | null>(null);
  const [loadingModalArticles, setLoadingModalArticles] = useState<boolean>(false);

  // Regional state structures
  const defaultStatesList: StateTrendingData[] = [
    {
      state: "Andhra Pradesh",
      code: "AP",
      trendingTopic: "Amaravati Infrastructure & Regional Development Projects",
      category: "Infrastructure",
      articleCount: 45,
      velocity: "+30% growth",
      description: "State news tracking capital city developments, Visakhapatnam port expansion, and regional economic reforms.",
      articles: []
    },
    {
      state: "Telangana",
      code: "TG",
      trendingTopic: "Hyderabad IT Hub, Metro Extensions & Technology Investments",
      category: "Technology",
      articleCount: 42,
      velocity: "+28% growth",
      description: "Regional intelligence tracking Cyberabad AI investments, Hyderabad metro transit, and technology clusters.",
      articles: []
    },
    {
      state: "Maharashtra",
      code: "MH",
      trendingTopic: "Mumbai Financial District & Port Infrastructure Projects",
      category: "Business",
      articleCount: 56,
      velocity: "+32% growth",
      description: "Business intelligence tracking Mumbai stock market trends, BKC developments, and Navi Mumbai airport.",
      articles: []
    },
    {
      state: "Karnataka",
      code: "KA",
      trendingTopic: "Bengaluru Tech Hub, Semiconductor Fabrications & R&D",
      category: "Technology",
      articleCount: 49,
      velocity: "+25% growth",
      description: "Technology news tracking Bengaluru microchip packaging, startup funding, and EV manufacturing hubs.",
      articles: []
    },
    {
      state: "Delhi NCR",
      code: "DL",
      trendingTopic: "Delhi Electric Fleet Extensions & Environmental Policy",
      category: "Environment",
      articleCount: 38,
      velocity: "+21% growth",
      description: "Capital news tracking electric bus transit extensions, municipal air quality policies, and urban updates.",
      articles: []
    },
    {
      state: "Tamil Nadu",
      code: "TN",
      trendingTopic: "Chennai Automotive Export Corridors & Clean Energy",
      category: "Manufacturing",
      articleCount: 35,
      velocity: "+19% growth",
      description: "Industrial news tracking Chennai EV export manufacturing, green hydrogen projects, and coastal logistics.",
      articles: []
    }
  ];

  // Fetch 15 articles for a state directly from Backend API
  const fetchLiveStateArticles = async (stateName: string) => {
    try {
      const res = await apiClient.get("/api/dashboard/latest", {
        params: { query: stateName, limit: 15 }
      });
      if (res.data && res.data.articles && res.data.articles.length > 0) {
        const fetchedArts: Article[] = res.data.articles.map((art: any, i: number) => ({
          id: art.id || `st_${stateName}_${i}`,
          title: art.title,
          source: art.source || "News Source",
          url: art.url || `https://news.google.com/search?q=${encodeURIComponent(art.title)}`,
          category: art.category || "Regional",
          published_date: art.published_date || new Date().toISOString(),
          summary: art.cleaned_content || art.content || art.title,
          content: art.cleaned_content || art.content || art.title,
          state: stateName
        }));
        setLiveStateArticles(prev => ({ ...prev, [stateName]: fetchedArts }));
        return fetchedArts;
      }
    } catch (err) {
      console.error(`Error fetching news for ${stateName}:`, err);
    }
    return null;
  };

  const loadAllStateFeeds = async () => {
    setLoadingStateNews(true);
    await Promise.all(defaultStatesList.map(st => fetchLiveStateArticles(st.state)));
    setLoadingStateNews(false);
  };

  useEffect(() => {
    loadAllStateFeeds();
  }, []);

  const stateNamesList = ["All", ...defaultStatesList.map(s => s.state)];

  const getStateDataWithLiveArticles = (stData: StateTrendingData): StateTrendingData => {
    const cached = liveStateArticles[stData.state];
    if (cached && cached.length > 0) {
      const topTitle = cached[0]?.title || stData.trendingTopic;
      const snippet = (cached[0]?.cleaned_content || cached[0]?.summary || cached[0]?.content || "").slice(0, 180) || stData.description;
      const count = cached.length;
      return {
        ...stData,
        trendingTopic: topTitle,
        description: snippet,
        articleCount: count,
        velocity: `+${Math.min(45, count * 3 + 12)}% growth`,
        articles: cached
      };
    }
    return stData;
  };


  const getCustomStateData = (query: string): StateTrendingData | null => {
    if (!query || query.trim().length === 0) return null;
    const cleanQuery = query.trim();
    const existing = defaultStatesList.find(s => s.state.toLowerCase() === cleanQuery.toLowerCase());
    if (existing) return getStateDataWithLiveArticles(existing);

    const stateName = cleanQuery.charAt(0).toUpperCase() + cleanQuery.slice(1);
    const code = stateName.substring(0, 2).toUpperCase();
    const cached = liveStateArticles[stateName] || [];
    const topTitle = cached[0]?.title || `${stateName} Economic Growth, Policy Updates & Infrastructure Projects`;

    return {
      state: stateName,
      code: code,
      trendingTopic: topTitle,
      category: "Regional News",
      articleCount: cached.length > 0 ? cached.length : 15,
      velocity: `+${Math.min(45, (cached.length || 5) * 3 + 10)}% growth`,
      description: `State news tracking regional developments, governance, and updates across ${stateName}.`,
      articles: cached
    };
  };


  const handleSearchSubmit = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    const query = stateSearchQuery.trim();
    setActiveSearch(query);
    if (query) {
      setSelectedState("All");
      setLoadingStateNews(true);
      const fetched = await fetchLiveStateArticles(query);
      setLoadingStateNews(false);
      
      const customMatch = getCustomStateData(query);
      if (customMatch && fetched) {
        customMatch.articles = fetched;
        setActiveStateModal(customMatch);
      }
    }
  };

  const displayStateList = defaultStatesList.map(st => getStateDataWithLiveArticles(st));
  if (activeSearch && !defaultStatesList.some(s => s.state.toLowerCase().includes(activeSearch.toLowerCase()))) {
    const custom = getCustomStateData(activeSearch);
    if (custom) displayStateList.unshift(custom);
  }

  const filteredStateTopics = displayStateList.filter(item => {
    const searchToUse = activeSearch || stateSearchQuery;
    const matchesSearch = !searchToUse || 
                          item.state.toLowerCase().includes(searchToUse.toLowerCase()) ||
                          item.trendingTopic.toLowerCase().includes(searchToUse.toLowerCase()) ||
                          item.category.toLowerCase().includes(searchToUse.toLowerCase());
    const matchesStateSelect = selectedState === "All" || item.state === selectedState;
    return matchesSearch && matchesStateSelect;
  });

  const handleOpenStateModal = async (stData: StateTrendingData) => {
    setActiveStateModal(stData);
    setLoadingModalArticles(true);
    let cached = liveStateArticles[stData.state];
    if (!cached || cached.length === 0) {
      cached = await fetchLiveStateArticles(stData.state) || [];
    }
    if (cached.length > 0) {
      setActiveStateModal(prev => prev ? { ...prev, articles: cached, articleCount: cached.length } : null);
    }
    setLoadingModalArticles(false);
  };



  return (
    <div className="space-y-8 max-w-7xl mx-auto pb-16">
      {/* Top Banner Header */}
      <div className="bg-gradient-to-r from-indigo-900 via-slate-900 to-indigo-950 text-white p-6 md:p-8 rounded-3xl shadow-xl flex flex-col md:flex-row items-start md:items-center justify-between gap-6 border border-slate-800">
        <div className="space-y-2 max-w-2xl">
          <div className="flex items-center gap-2">
            <span className="p-2 bg-indigo-500/20 rounded-xl text-indigo-300 border border-indigo-500/30">
              <MapPin className="w-5 h-5 text-indigo-400" />
            </span>
            <span className="text-xs font-extrabold uppercase tracking-widest text-indigo-300">
              Regional News Intelligence
            </span>
          </div>
          <h1 className="text-3xl font-black text-white tracking-tight">
            State-Wise Trending Topics
          </h1>
          <p className="text-slate-300 text-xs leading-relaxed">
            Type any state name and press <strong>Enter</strong> to view real-time state articles and developments.
          </p>
        </div>

        {/* Refresh & Search Form */}
        <div className="flex flex-col md:flex-row items-stretch md:items-center gap-3 w-full md:w-auto">
          <button
            onClick={loadAllStateFeeds}
            disabled={loadingStateNews}
            className="px-4 py-2.5 bg-indigo-600 hover:bg-indigo-700 text-white font-extrabold text-xs rounded-xl transition-all cursor-pointer shadow-md flex items-center justify-center gap-2 shrink-0 disabled:opacity-50"
          >
            <RefreshCw className={`w-4 h-4 ${loadingStateNews ? "animate-spin" : ""}`} />
            {loadingStateNews ? "Refreshing..." : "Refresh Feeds"}
          </button>

          <form onSubmit={handleSearchSubmit} className="space-y-1 shrink-0">
            <div className="relative flex items-center gap-2">
              <div className="relative flex-1">
                <Search className="w-4 h-4 text-slate-400 absolute left-3.5 top-3" />
                <input
                  type="text"
                  placeholder="Type state (e.g. Gujarat, Kerala...)"
                  value={stateSearchQuery}
                  onChange={(e) => {
                    setStateSearchQuery(e.target.value);
                    setActiveSearch(e.target.value);
                    setSelectedState("All");
                  }}
                  className="w-full pl-10 pr-4 py-2.5 bg-slate-800/90 text-white placeholder-slate-400 border border-slate-700 rounded-xl text-xs focus:outline-none focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-400 transition-all shadow-inner"
                />
              </div>
              <button
                type="submit"
                className="px-4 py-2.5 bg-indigo-600 hover:bg-indigo-700 text-white font-extrabold text-xs rounded-xl transition-all cursor-pointer shadow-md shrink-0 flex items-center gap-1"
              >
                Search <ArrowRight className="w-3.5 h-3.5" />
              </button>
            </div>
          </form>
        </div>
      </div>

      {/* State Quick-Filter Pills */}
      <div className="bg-white p-4 rounded-2xl border border-slate-200/80 shadow-xs flex items-center gap-2 overflow-x-auto">
        <span className="text-xs font-extrabold text-slate-400 uppercase tracking-wider flex items-center gap-1.5 mr-2 shrink-0">
          <Building2 className="w-4 h-4 text-indigo-600" /> Select Region:
        </span>
        {stateNamesList.map((st) => (
          <button
            key={st}
            onClick={() => {
              setSelectedState(st);
              setStateSearchQuery("");
              setActiveSearch("");
              if (st !== "All") fetchLiveStateArticles(st);
            }}
            className={`px-4 py-2 rounded-xl text-xs font-extrabold transition-all cursor-pointer whitespace-nowrap ${
              selectedState === st && !stateSearchQuery && !activeSearch
                ? "bg-indigo-600 text-white shadow-md shadow-indigo-600/30"
                : "bg-slate-100 text-slate-700 hover:bg-slate-200 border border-slate-200/80"
            }`}
          >
            {st}
          </button>
        ))}
      </div>

      {/* State Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {filteredStateTopics.map((stData) => (
          <div 
            key={stData.state}
            className="bg-white p-6 rounded-3xl border border-slate-200/90 shadow-xs hover:shadow-lg hover:border-indigo-300 transition-all space-y-4 flex flex-col justify-between group"
          >
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="px-3 py-1 bg-indigo-50 text-indigo-700 font-black text-xs rounded-xl border border-indigo-100 flex items-center gap-1.5 shadow-2xs">
                  <MapPin className="w-3.5 h-3.5 text-indigo-600" />
                  {stData.state} ({stData.code})
                </span>
                <span className="text-[10px] font-extrabold text-emerald-700 bg-emerald-50 px-2.5 py-1 rounded-md border border-emerald-100">
                  {stData.velocity}
                </span>
              </div>

              <h3 
                onClick={() => handleOpenStateModal(stData)}
                className="text-base font-extrabold text-slate-900 leading-snug group-hover:text-indigo-600 transition-colors cursor-pointer"
              >
                {stData.trendingTopic}
              </h3>

              <p className="text-xs text-slate-500 line-clamp-2 leading-relaxed">
                {stData.description}
              </p>

              <div className="flex items-center justify-between text-xs pt-2">
                <span className="text-slate-500 font-medium">Article Volume:</span>
                <span className="font-extrabold text-indigo-900 bg-indigo-50 px-2.5 py-1 rounded-lg border border-indigo-100">
                  {stData.articles.length > 0 ? `${stData.articles.length} Articles` : `${stData.articleCount} Articles`}
                </span>
              </div>
            </div>

            <div className="pt-3 border-t border-slate-100 flex items-center justify-between text-xs">
              <span className="text-slate-400 text-[11px] font-bold">State Intelligence</span>
              
              <button
                onClick={() => handleOpenStateModal(stData)}
                className="px-3.5 py-2 bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-extrabold rounded-xl border border-indigo-500 flex items-center gap-1.5 transition-all cursor-pointer shadow-md hover:shadow-lg active:scale-95"
              >
                View State Articles <ExternalLink className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>
        ))}

        {filteredStateTopics.length === 0 && (
          <div className="col-span-full bg-white p-12 rounded-3xl border border-slate-200 text-center space-y-3">
            <Compass className="w-10 h-10 text-slate-300 mx-auto" />
            <h4 className="text-base font-extrabold text-slate-800">No state news found for "{activeSearch || stateSearchQuery || selectedState}"</h4>
            <p className="text-xs text-slate-500">Try searching for Andhra Pradesh, Telangana, Maharashtra, Karnataka, Delhi NCR, or Tamil Nadu.</p>
          </div>
        )}
      </div>

      {/* REUSABLE STATE ARTICLE CLUSTER MODAL */}
      <ArticleModal
        isOpen={Boolean(activeStateModal)}
        onClose={() => setActiveStateModal(null)}
        title={activeStateModal?.trendingTopic || activeStateModal?.state || ""}
        subtitle={`Live State Intelligence & News Feed for ${activeStateModal?.state}`}
        category={activeStateModal?.category}
        badgeText={activeStateModal?.velocity}
        description={activeStateModal?.description}
        articles={activeStateModal?.articles || []}
        loading={loadingModalArticles}
        accentColor="indigo"
        onAskAI={(topicName) => {
          navigate("/chat", { state: { query: `Tell me about the trending state news in ${activeStateModal?.state}: ${topicName}` } });
        }}
      />
    </div>
  );
}
