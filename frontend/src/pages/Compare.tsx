import { useState, useEffect } from "react";
import { GitCompare, ExternalLink, RefreshCw, Sparkles, CheckCircle, Flame } from "lucide-react";
import apiClient from "../api/client";


interface CommonItem {
  title: string;
  summary: string;
  source1_title?: string;
  source2_title?: string;
  source1_url?: string;
  source2_url?: string;
}

interface ExclusiveItem {
  title: string;
  url?: string;
  category?: string;
}

interface ComparisonResult {
  source1: string;
  source2: string;
  source1_count?: number;
  source2_count?: number;
  common_news: CommonItem[];
  exclusive_source1: ExclusiveItem[];
  exclusive_source2: ExclusiveItem[];
}

export default function Compare() {
  const [sources, setSources] = useState<string[]>([]);
  const [source1, setSource1] = useState<string>("Times of India");
  const [source2, setSource2] = useState<string>("The Hindu");

  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<ComparisonResult | null>(null);

  const fetchSources = async () => {
    try {
      const res = await apiClient.get("/api/sources/");
      const srcList = res.data.sources || [];
      setSources(srcList);
      if (srcList.length >= 2) {
        setSource1(srcList[0]);
        setSource2(srcList[1]);
        handleAnalyze(srcList[0], srcList[1]);
      } else {
        handleAnalyze("Times of India", "The Hindu");
      }
    } catch (err) {
      console.error("Error fetching sources:", err);
      handleAnalyze("Times of India", "The Hindu");
    }
  };

  const handleAnalyze = async (s1 = source1, s2 = source2) => {
    setLoading(true);
    try {
      const res = await apiClient.post("/api/compare/", {
        source1: s1,
        source2: s2,
      });
      setResult(res.data);
    } catch (err) {
      console.error("Error running source comparison:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchSources();
  }, []);

  return (
    <div className="space-y-8 max-w-6xl mx-auto">
      {/* Header */}
      <div className="bg-white p-6 md:p-8 rounded-2xl shadow-xs border border-slate-200/80 space-y-4">
        <div className="flex items-center gap-3">
          <div className="p-3 bg-indigo-50 text-indigo-600 rounded-xl">
            <GitCompare className="w-6 h-6" />
          </div>
          <div>
            <h1 className="text-2xl font-extrabold text-slate-900 tracking-tight">
              Compare News Outlet Coverage
            </h1>
            <p className="text-xs text-slate-500 mt-0.5">
              Analyze overlapping topics and exclusive reporting differences between major media outlets in real-time.
            </p>
          </div>
        </div>

        {/* Control Bar */}
        <div className="pt-4 border-t border-slate-100 flex flex-col sm:flex-row items-center gap-4">
          <div className="flex-1 w-full">
            <label className="block text-xs font-bold text-slate-500 uppercase tracking-wider mb-1.5">
              Source Outlet 1
            </label>
            <select
              value={source1}
              onChange={(e) => setSource1(e.target.value)}
              className="w-full p-3 bg-slate-50 border border-slate-200 rounded-xl text-sm font-semibold text-slate-800 focus:outline-none focus:ring-2 focus:ring-indigo-500 cursor-pointer"
            >
              {sources.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </div>

          <div className="text-xl font-black text-indigo-400 mt-4 sm:mt-6">VS</div>

          <div className="flex-1 w-full">
            <label className="block text-xs font-bold text-slate-500 uppercase tracking-wider mb-1.5">
              Source Outlet 2
            </label>
            <select
              value={source2}
              onChange={(e) => setSource2(e.target.value)}
              className="w-full p-3 bg-slate-50 border border-slate-200 rounded-xl text-sm font-semibold text-slate-800 focus:outline-none focus:ring-2 focus:ring-indigo-500 cursor-pointer"
            >
              {sources.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </div>

          <div className="w-full sm:w-auto mt-4 sm:mt-6">
            <button
              onClick={() => handleAnalyze(source1, source2)}
              disabled={loading || source1 === source2}
              className="w-full sm:w-auto px-6 py-3 bg-indigo-600 hover:bg-indigo-700 disabled:bg-indigo-300 text-white rounded-xl text-xs font-bold shadow-sm transition-all flex items-center justify-center gap-2 cursor-pointer"
            >
              {loading ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  Analyzing...
                </>
              ) : (
                <>
                  <Sparkles className="w-4 h-4" />
                  Run Media Analysis
                </>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Comparison Results */}
      {result && (
        <div className="space-y-6">
          {/* Outlet Stats Header */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
            <div className="bg-gradient-to-r from-blue-900 to-indigo-900 text-white p-5 rounded-2xl shadow-md border border-blue-800 flex items-center justify-between">
              <div>
                <span className="text-xs text-blue-200 font-semibold uppercase">Source Outlet A</span>
                <h3 className="text-lg font-bold mt-0.5">{result.source1}</h3>
              </div>
              <div className="text-right">
                <span className="text-2xl font-black">{result.source1_count || 0}</span>
                <span className="block text-[10px] text-blue-200">Articles Indexed</span>
              </div>
            </div>

            <div className="bg-gradient-to-r from-violet-900 to-purple-900 text-white p-5 rounded-2xl shadow-md border border-violet-800 flex items-center justify-between">
              <div>
                <span className="text-xs text-violet-200 font-semibold uppercase">Source Outlet B</span>
                <h3 className="text-lg font-bold mt-0.5">{result.source2}</h3>
              </div>
              <div className="text-right">
                <span className="text-2xl font-black">{result.source2_count || 0}</span>
                <span className="block text-[10px] text-violet-200">Articles Indexed</span>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Common News Coverage */}
            <div className="bg-white p-6 rounded-2xl shadow-xs border border-slate-200/80 space-y-4">
              <h2 className="text-base font-bold text-slate-900 flex items-center gap-2 border-b border-slate-100 pb-3">
                <CheckCircle className="w-5 h-5 text-emerald-600" />
                Overlapping News Topics ({result.common_news.length})
              </h2>

              <div className="space-y-4">
                {result.common_news.map((item, idx) => (
                  <div key={idx} className="p-4 bg-slate-50 rounded-xl border border-slate-200/60 space-y-2">
                    <h4 className="font-bold text-sm text-slate-900">{item.title}</h4>
                    <p className="text-xs text-slate-600">{item.summary}</p>
                    <div className="pt-2 flex flex-col gap-1.5 text-xs">
                      {item.source1_title && (
                        <a
                          href={item.source1_url || "#"}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-blue-700 hover:underline flex items-center gap-1 font-medium"
                        >
                          <span className="font-bold text-[10px] uppercase px-1.5 py-0.5 bg-blue-100 rounded text-blue-800">
                            {result.source1}
                          </span>
                          <span className="truncate">{item.source1_title}</span>
                          <ExternalLink className="w-3 h-3 shrink-0" />
                        </a>
                      )}
                      {item.source2_title && (
                        <a
                          href={item.source2_url || "#"}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-violet-700 hover:underline flex items-center gap-1 font-medium"
                        >
                          <span className="font-bold text-[10px] uppercase px-1.5 py-0.5 bg-violet-100 rounded text-violet-800">
                            {result.source2}
                          </span>
                          <span className="truncate">{item.source2_title}</span>
                          <ExternalLink className="w-3 h-3 shrink-0" />
                        </a>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Exclusive Coverage */}
            <div className="bg-white p-6 rounded-2xl shadow-xs border border-slate-200/80 space-y-4">
              <h2 className="text-base font-bold text-slate-900 flex items-center gap-2 border-b border-slate-100 pb-3">
                <Flame className="w-5 h-5 text-amber-600" />
                Exclusive / Distinct Headlines
              </h2>

              <div className="space-y-4">
                {/* Source 1 Exclusive */}
                <div className="space-y-2">
                  <span className="text-xs font-bold text-blue-700 uppercase tracking-wider block">
                    Exclusive to {result.source1}
                  </span>
                  {result.exclusive_source1.length === 0 ? (
                    <p className="text-xs text-slate-400 italic">No exclusive articles detected.</p>
                  ) : (
                    result.exclusive_source1.map((item, idx) => (
                      <a
                        key={idx}
                        href={item.url || "#"}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="block p-3 bg-blue-50/60 hover:bg-blue-100/70 rounded-xl border border-blue-200/60 transition-colors group cursor-pointer"
                      >
                        <div className="flex items-center justify-between gap-2">
                          <h5 className="text-xs font-bold text-slate-900 group-hover:text-blue-700 transition-colors line-clamp-1">
                            {item.title}
                          </h5>
                          <ExternalLink className="w-3.5 h-3.5 text-blue-500 shrink-0" />
                        </div>
                        {item.category && (
                          <span className="text-[10px] text-blue-600 font-semibold mt-1 block">
                            Category: {item.category}
                          </span>
                        )}
                      </a>
                    ))
                  )}
                </div>

                {/* Source 2 Exclusive */}
                <div className="space-y-2 pt-2 border-t border-slate-100">
                  <span className="text-xs font-bold text-violet-700 uppercase tracking-wider block">
                    Exclusive to {result.source2}
                  </span>
                  {result.exclusive_source2.length === 0 ? (
                    <p className="text-xs text-slate-400 italic">No exclusive articles detected.</p>
                  ) : (
                    result.exclusive_source2.map((item, idx) => (
                      <a
                        key={idx}
                        href={item.url || "#"}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="block p-3 bg-violet-50/60 hover:bg-violet-100/70 rounded-xl border border-violet-200/60 transition-colors group cursor-pointer"
                      >
                        <div className="flex items-center justify-between gap-2">
                          <h5 className="text-xs font-bold text-slate-900 group-hover:text-violet-700 transition-colors line-clamp-1">
                            {item.title}
                          </h5>
                          <ExternalLink className="w-3.5 h-3.5 text-violet-500 shrink-0" />
                        </div>
                        {item.category && (
                          <span className="text-[10px] text-violet-600 font-semibold mt-1 block">
                            Category: {item.category}
                          </span>
                        )}
                      </a>
                    ))
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
