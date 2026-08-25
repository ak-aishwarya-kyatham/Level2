import { useState, useEffect } from "react";
import { 
  BarChart3, 
  Layers, 
  Globe, 
  RefreshCw, 
  TrendingUp, 
  ShieldAlert, 
  Cpu, 
  FileText, 
  CheckCircle2, 
  Activity, 
  Clock, 
  PieChart, 
  Scale, 
  Sparkles, 
  Database, 
  BrainCircuit
} from "lucide-react";
import apiClient from "../api/client";
import MetricCard from "../components/MetricCard";


interface CategoryMetric {
  name: string;
  percentage: number;
  count: number;
}

interface SourceMetric {
  name: string;
  percentage: number;
  count: number;
}

interface SentimentMetric {
  tone: string;
  percentage: number;
  count: number;
}

interface EntityMetric {
  entity: string;
  percentage: number;
  count: number;
}

interface TimelineMetric {
  time: string;
  count: number;
}

interface AgentPerformanceMetric {
  agent: string;
  avg_latency_ms: number;
  success_rate: number;
}

interface WordLengthMetric {
  range: string;
  percentage: number;
  count: number;
}

interface AnalyticsData {
  categories: CategoryMetric[];
  sources: SourceMetric[];
  sentiment?: SentimentMetric[];
  top_entities?: EntityMetric[];
  timeline?: TimelineMetric[];
  agent_performance?: AgentPerformanceMetric[];
  word_length?: WordLengthMetric[];
  total_articles?: number;
  duplicates_avoided?: number;
}

export default function Analytics() {
  const [data, setData] = useState<AnalyticsData | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [activeTab, setActiveTab] = useState<"all" | "distribution" | "sentiment" | "pipeline">("all");

  const fetchMetrics = async () => {
    setLoading(true);
    try {
      const res = await apiClient.get("/api/analytics/metrics");
      setData(res.data);
    } catch (err) {
      console.error("Error fetching analytics metrics:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchMetrics();
    window.addEventListener("liveNewsRefreshed", fetchMetrics);
    return () => window.removeEventListener("liveNewsRefreshed", fetchMetrics);
  }, []);

  const categoryColors = [
    "bg-indigo-600",
    "bg-blue-600",
    "bg-emerald-600",
    "bg-amber-500",
    "bg-rose-500",
    "bg-violet-600",
    "bg-cyan-600",
    "bg-slate-600"
  ];

  const getToneStyle = (tone: string) => {
    if (tone.includes("Critical")) return { color: "text-rose-600", bg: "bg-rose-50", border: "border-rose-200", bar: "bg-rose-500", icon: ShieldAlert };
    if (tone.includes("Market")) return { color: "text-emerald-600", bg: "bg-emerald-50", border: "border-emerald-200", bar: "bg-emerald-500", icon: TrendingUp };
    if (tone.includes("Policy")) return { color: "text-purple-600", bg: "bg-purple-50", border: "border-purple-200", bar: "bg-purple-500", icon: Scale };
    return { color: "text-blue-600", bg: "bg-blue-50", border: "border-blue-200", bar: "bg-blue-500", icon: FileText };
  };

  const getEntityStyle = (idx: number) => {
    const styles = [
      { color: "text-indigo-600", bg: "bg-indigo-50", bar: "bg-indigo-600" },
      { color: "text-cyan-600", bg: "bg-cyan-50", bar: "bg-cyan-600" },
      { color: "text-amber-600", bg: "bg-amber-50", bar: "bg-amber-600" },
      { color: "text-emerald-600", bg: "bg-emerald-50", bar: "bg-emerald-600" },
    ];
    return styles[idx % styles.length];
  };

  // Calculate timeline max value for proportioning bars
  const maxTimelineCount = data?.timeline 
    ? Math.max(...data.timeline.map((t) => t.count), 1) 
    : 100;

  // Calculate total agent latency from real backend data
  const totalLatency = data?.agent_performance && data.agent_performance.length > 0 
    ? data.agent_performance.reduce((acc, curr) => acc + curr.avg_latency_ms, 0)
    : null;

  const deduplicationSavings = data?.total_articles && data?.duplicates_avoided
    ? Math.round((data.duplicates_avoided / (data.total_articles + data.duplicates_avoided)) * 100)
    : null;

  const categoryConfidence = null;


  return (
    <div className="space-y-8 max-w-6xl mx-auto pb-12">
      {/* Page Header */}
      <div className="bg-white p-6 md:p-8 rounded-2xl shadow-xs border border-slate-200/80 flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div className="flex items-center gap-3.5">
          <div className="p-3 bg-indigo-600 text-white rounded-xl shadow-md shadow-indigo-200">
            <BarChart3 className="w-6 h-6" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-2xl font-extrabold text-slate-900 tracking-tight">
                Media Analytics & Trends
              </h1>
              <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[10px] font-bold bg-emerald-100 text-emerald-800 border border-emerald-200">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-ping" />
                LIVE ACCUMULATION
              </span>
            </div>
            <p className="text-xs text-slate-500 mt-1">
              Real-time statistics breakdown across news feed velocity, sentiment radar, entity intelligence, and multi-agent AI pipeline health.
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3 w-full md:w-auto justify-between md:justify-end">
          {/* Refresh Button */}
          <button
            onClick={fetchMetrics}
            disabled={loading}
            className="px-4 py-2.5 bg-indigo-50 hover:bg-indigo-100 text-indigo-700 text-xs font-bold rounded-xl transition-all duration-200 flex items-center gap-2 cursor-pointer border border-indigo-100 shadow-2xs active:scale-95"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin" : ""}`} />
            Refresh Analytics
          </button>
        </div>
      </div>

      {/* KPI Overview Summary Bar */}
      {data && (
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <MetricCard
            title="Total Index Corpus"
            value={data.total_articles || 0}
            subtitle="Verified news articles indexed"
            icon={Database}
            iconBgColor="bg-indigo-50"
            iconTextColor="text-indigo-600"
            badgeText="Full Corpus"
            badgeColor="indigo"
          />

          <MetricCard
            title="Deduplication Efficiency"
            value={deduplicationSavings !== null ? `${deduplicationSavings}%` : "N/A"}
            subtitle={deduplicationSavings !== null ? "Article volume saved via semantic dedup" : "[Sample Data / Awaiting Run]"}
            icon={Sparkles}
            iconBgColor="bg-emerald-50"
            iconTextColor="text-emerald-600"
            badgeText={deduplicationSavings !== null ? `${deduplicationSavings}% Saved` : "Sample Data"}
            badgeColor={deduplicationSavings !== null ? "emerald" : "amber"}
          />

          <MetricCard
            title="Pipeline Latency"
            value={totalLatency !== null ? `${totalLatency}ms` : "N/A"}
            subtitle={totalLatency !== null ? "End-to-end multi-agent pipeline wall clock" : "[Sample Data / Awaiting Run]"}
            icon={BrainCircuit}
            iconBgColor="bg-violet-50"
            iconTextColor="text-violet-600"
            badgeText={totalLatency !== null ? "Optimized" : "Sample Data"}
            badgeColor={totalLatency !== null ? "indigo" : "amber"}
          />

          <MetricCard
            title="Categorization Quality"
            value={categoryConfidence !== null ? `${categoryConfidence}%` : "N/A"}
            subtitle={categoryConfidence !== null ? "AI entity classification score" : "[Sample Data / Awaiting Run]"}
            icon={ShieldAlert}
            iconBgColor="bg-emerald-50"
            iconTextColor="text-emerald-600"
            badgeText={categoryConfidence !== null ? "High Precision" : "Sample Data"}
            badgeColor={categoryConfidence !== null ? "emerald" : "amber"}
          />
        </div>
      )}


      {/* Analytics View Selector Navigation Tabs */}
      <div className="flex items-center gap-2 border-b border-slate-200 pb-2 overflow-x-auto">
        <button
          onClick={() => setActiveTab("all")}
          className={`px-4 py-2 text-xs font-bold rounded-xl transition-colors cursor-pointer flex items-center gap-1.5 ${
            activeTab === "all"
              ? "bg-slate-900 text-white shadow-xs"
              : "bg-white text-slate-600 hover:bg-slate-100 border border-slate-200"
          }`}
        >
          <BarChart3 className="w-3.5 h-3.5" />
          All Visual Analytics
        </button>
        <button
          onClick={() => setActiveTab("distribution")}
          className={`px-4 py-2 text-xs font-bold rounded-xl transition-colors cursor-pointer flex items-center gap-1.5 ${
            activeTab === "distribution"
              ? "bg-slate-900 text-white shadow-xs"
              : "bg-white text-slate-600 hover:bg-slate-100 border border-slate-200"
          }`}
        >
          <Layers className="w-3.5 h-3.5" />
          Categories & Publishers
        </button>
        <button
          onClick={() => setActiveTab("sentiment")}
          className={`px-4 py-2 text-xs font-bold rounded-xl transition-colors cursor-pointer flex items-center gap-1.5 ${
            activeTab === "sentiment"
              ? "bg-slate-900 text-white shadow-xs"
              : "bg-white text-slate-600 hover:bg-slate-100 border border-slate-200"
          }`}
        >
          <PieChart className="w-3.5 h-3.5" />
          Sentiment & Entity Frequency
        </button>
        <button
          onClick={() => setActiveTab("pipeline")}
          className={`px-4 py-2 text-xs font-bold rounded-xl transition-colors cursor-pointer flex items-center gap-1.5 ${
            activeTab === "pipeline"
              ? "bg-slate-900 text-white shadow-xs"
              : "bg-white text-slate-600 hover:bg-slate-100 border border-slate-200"
          }`}
        >
          <Cpu className="w-3.5 h-3.5" />
          Pipeline & Velocity
        </button>
      </div>

      {loading ? (
        <div className="bg-white p-16 rounded-2xl text-center shadow-xs border border-slate-200/80 space-y-3">
          <RefreshCw className="w-8 h-8 text-indigo-600 animate-spin mx-auto" />
          <p className="text-xs font-semibold text-slate-600">Calculating platform intelligence and trends from stored feeds...</p>
        </div>
      ) : data ? (
        <div className="space-y-8">
          {/* SECTION 1: 24-HOUR FEED VELOCITY TIMELINE (GRAPH 1) */}
          {(activeTab === "all" || activeTab === "pipeline") && data.timeline && (
            <div className="bg-white p-6 md:p-8 rounded-2xl shadow-xs border border-slate-200/80 space-y-6">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-100 pb-4">
                <div>
                  <h2 className="text-base font-bold text-slate-900 flex items-center gap-2">
                    <Clock className="w-5 h-5 text-indigo-600" />
                    24-Hour Feed Velocity & Ingestion Timeline
                  </h2>
                  <p className="text-xs text-slate-500 mt-0.5">
                    Distribution of incoming articles processed across 4-hour operational windows.
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  <span className="px-2.5 py-1 rounded-md bg-indigo-50 text-indigo-700 text-[11px] font-bold border border-indigo-100">
                    Peak Interval: {data.timeline.reduce((max, item) => item.count > max.count ? item : max, data.timeline[0])?.time || "08:00 - 12:00"}
                  </span>
                </div>
              </div>


              {/* Bar Chart Visualization */}
              <div className="pt-6 pb-2">
                <div className="h-48 flex items-end justify-between gap-3 md:gap-6 px-2">
                  {data.timeline.map((item, idx) => {
                    const heightPercent = Math.max(Math.round((item.count / maxTimelineCount) * 100), 12);
                    const isPeak = item.count === maxTimelineCount;

                    return (
                      <div key={idx} className="flex-1 flex flex-col items-center gap-2 group h-full justify-end">
                        {/* Tooltip on hover */}
                        <div className="text-[11px] font-extrabold text-slate-700 group-hover:text-indigo-600 transition-colors mb-1">
                          {item.count} <span className="text-[9px] font-normal text-slate-400">arts</span>
                        </div>

                        {/* Bar */}
                        <div className="w-full max-w-[56px] bg-slate-100 rounded-t-xl overflow-hidden relative flex items-end h-full">
                          <div
                            className={`w-full rounded-t-xl transition-all duration-700 ease-out group-hover:brightness-110 ${
                              isPeak 
                                ? "bg-gradient-to-t from-indigo-700 to-indigo-500 shadow-md shadow-indigo-200" 
                                : "bg-gradient-to-t from-slate-700 to-slate-500"
                            }`}
                            style={{ height: `${heightPercent}%` }}
                          >
                            {isPeak && (
                              <div className="absolute top-1 left-1/2 -translate-x-1/2 w-1.5 h-1.5 bg-amber-300 rounded-full animate-ping" />
                            )}
                          </div>
                        </div>

                        {/* Time Label */}
                        <span className={`text-[10px] font-bold text-center mt-2 group-hover:text-indigo-600 transition-colors ${
                          isPeak ? "text-indigo-600" : "text-slate-500"
                        }`}>
                          {item.time}
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          )}

          {/* SECTION 2: SENTIMENT RADAR & ENTITY FREQUENCY (GRAPHS 2 & 3) */}
          {(activeTab === "all" || activeTab === "sentiment") && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              {/* GRAPH 2: MEDIA SENTIMENT & TONE ANALYSIS */}
              {data.sentiment && (
                <div className="bg-white p-6 rounded-2xl shadow-xs border border-slate-200/80 space-y-6 flex flex-col justify-between">
                  <div>
                    <div className="flex items-center justify-between border-b border-slate-100 pb-3">
                      <h2 className="text-base font-bold text-slate-900 flex items-center gap-2">
                        <PieChart className="w-5 h-5 text-rose-500" />
                        Media Sentiment & Tone Radar
                      </h2>
                      <span className="text-xs text-slate-400 font-semibold">4 Heuristic Tone Buckets</span>
                    </div>
                    <p className="text-xs text-slate-500 mt-2">
                      Automated classification of article headlines and text into editorial tone categories.
                    </p>
                  </div>

                  {/* Multi-Tone Progress Stack */}
                  <div className="space-y-4">
                    <div className="h-4 w-full bg-slate-100 rounded-full flex overflow-hidden p-0.5 border border-slate-200">
                      {data.sentiment.map((s) => {
                        const style = getToneStyle(s.tone);
                        return (
                          <div
                            key={s.tone}
                            title={`${s.tone}: ${s.percentage}%`}
                            className={`${style.bar} h-full first:rounded-l-full last:rounded-r-full transition-all duration-500 hover:opacity-90`}
                            style={{ width: `${Math.max(s.percentage, 4)}%` }}
                          />
                        );
                      })}
                    </div>

                    {/* Tone Cards List */}
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-2">
                      {data.sentiment.map((s) => {
                        const style = getToneStyle(s.tone);
                        const Icon = style.icon;
                        return (
                          <div key={s.tone} className={`p-3.5 rounded-xl border ${style.bg} ${style.border} space-y-1.5`}>
                            <div className="flex items-center justify-between">
                              <span className={`text-xs font-bold ${style.color} flex items-center gap-1.5`}>
                                <Icon className="w-3.5 h-3.5" />
                                {s.tone}
                              </span>
                              <span className="text-xs font-black text-slate-800">{s.percentage}%</span>
                            </div>
                            <div className="flex justify-between items-center text-[11px] text-slate-500">
                              <span>Coverage Volume</span>
                              <span className="font-semibold text-slate-700">{s.count} articles</span>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </div>
              )}

              {/* GRAPH 3: TOP ENTITY & DOMAIN FREQUENCY */}
              {data.top_entities && (
                <div className="bg-white p-6 rounded-2xl shadow-xs border border-slate-200/80 space-y-6 flex flex-col justify-between">
                  <div>
                    <div className="flex items-center justify-between border-b border-slate-100 pb-3">
                      <h2 className="text-base font-bold text-slate-900 flex items-center gap-2">
                        <Activity className="w-5 h-5 text-indigo-600" />
                        Key Domain & Entity Intelligence
                      </h2>
                      <span className="text-xs text-slate-400 font-semibold">Mention Density</span>
                    </div>
                    <p className="text-xs text-slate-500 mt-2">
                      Frequency of high-impact subject matter clusters detected across the news stream.
                    </p>
                  </div>

                  <div className="space-y-4">
                    {data.top_entities.map((item, idx) => {
                      const style = getEntityStyle(idx);
                      return (
                        <div key={item.entity} className="space-y-1.5">
                          <div className="flex justify-between text-xs font-semibold">
                            <span className="text-slate-800 flex items-center gap-1.5">
                              <span className={`w-2 h-2 rounded-full ${style.bar}`} />
                              {item.entity}
                            </span>
                            <span className="text-slate-500">
                              <span className="font-bold text-slate-800">{item.percentage}%</span>{" "}
                              <span className="text-slate-400 font-normal">({item.count} articles)</span>
                            </span>
                          </div>
                          <div className="w-full bg-slate-100 rounded-full h-3 overflow-hidden">
                            <div
                              className={`${style.bar} h-3 rounded-full transition-all duration-500`}
                              style={{ width: `${Math.max(item.percentage, 4)}%` }}
                            />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* SECTION 3: CATEGORY & PUBLISHER DISTRIBUTION (GRAPHS 4 & 5) */}
          {(activeTab === "all" || activeTab === "distribution") && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              {/* GRAPH 4: CATEGORY BREAKDOWN */}
              <div className="bg-white p-6 rounded-2xl shadow-xs border border-slate-200/80 space-y-6">
                <div className="flex items-center justify-between border-b border-slate-100 pb-3">
                  <h2 className="text-base font-bold text-slate-900 flex items-center gap-2">
                    <Layers className="w-5 h-5 text-indigo-600" />
                    Category Breakdown
                  </h2>
                  <span className="text-xs text-slate-400 font-semibold">{data.categories.length} Categories</span>
                </div>

                <div className="space-y-4 max-h-[380px] overflow-y-auto pr-1">
                  {data.categories.map((cat, idx) => {
                    const color = categoryColors[idx % categoryColors.length];
                    return (
                      <div key={cat.name} className="space-y-1.5">
                        <div className="flex justify-between text-xs font-semibold">
                          <span className="text-slate-800">{cat.name}</span>
                          <span className="text-slate-500">
                            {cat.percentage}% <span className="text-slate-400 font-normal">({cat.count} articles)</span>
                          </span>
                        </div>
                        <div className="w-full bg-slate-100 rounded-full h-3 overflow-hidden">
                          <div
                            className={`${color} h-3 rounded-full transition-all duration-500`}
                            style={{ width: `${Math.max(cat.percentage, 3)}%` }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* GRAPH 5: PUBLISHER SHARE */}
              <div className="bg-white p-6 rounded-2xl shadow-xs border border-slate-200/80 space-y-6">
                <div className="flex items-center justify-between border-b border-slate-100 pb-3">
                  <h2 className="text-base font-bold text-slate-900 flex items-center gap-2">
                    <Globe className="w-5 h-5 text-blue-600" />
                    Publisher Share & Media Distribution
                  </h2>
                  <span className="text-xs text-slate-400 font-semibold">{data.sources.length} Active Outlets</span>
                </div>

                <div className="space-y-4 max-h-[380px] overflow-y-auto pr-1">
                  {data.sources.map((src, idx) => {
                    const color = categoryColors[(idx + 2) % categoryColors.length];
                    return (
                      <div key={src.name} className="space-y-1.5">
                        <div className="flex justify-between text-xs font-semibold">
                          <span className="text-slate-800">{src.name}</span>
                          <span className="text-slate-500">
                            {src.percentage}% <span className="text-slate-400 font-normal">({src.count} articles)</span>
                          </span>
                        </div>
                        <div className="w-full bg-slate-100 rounded-full h-3 overflow-hidden">
                          <div
                            className={`${color} h-3 rounded-full transition-all duration-500`}
                            style={{ width: `${Math.max(src.percentage, 3)}%` }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          )}

          {/* SECTION 4: MULTI-AGENT PIPELINE PERFORMANCE & WORD LENGTH HISTOGRAM (GRAPHS 6 & 7) */}
          {(activeTab === "all" || activeTab === "pipeline") && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              {/* GRAPH 6: MULTI-AGENT PIPELINE HEALTH & LATENCY */}
              {data.agent_performance && (
                <div className="bg-white p-6 rounded-2xl shadow-xs border border-slate-200/80 space-y-6">
                  <div className="flex items-center justify-between border-b border-slate-100 pb-3">
                    <h2 className="text-base font-bold text-slate-900 flex items-center gap-2">
                      <Cpu className="w-5 h-5 text-violet-600" />
                      Multi-Agent Pipeline Health & Execution Latency
                    </h2>
                    <span className="text-xs font-bold text-indigo-600 bg-indigo-50 px-2 py-0.5 rounded-md border border-indigo-200">
                      Dynamic Pipeline Metrics
                    </span>
                  </div>
                  <p className="text-xs text-slate-500">
                    Execution time benchmark (ms) and success rate across backend agents calculated from actual execution logs.
                  </p>

                  <div className="space-y-4">
                    {data.agent_performance.map((agent: any) => (
                      <div key={agent.agent} className="p-3 bg-slate-50 rounded-xl border border-slate-200/60 space-y-2">
                        <div className="flex items-center justify-between text-xs">
                          <span className="font-bold text-slate-800 flex items-center gap-1.5">
                            <CheckCircle2 className={`w-3.5 h-3.5 ${agent.success_rate != null ? "text-emerald-500" : "text-slate-400"}`} />
                            {agent.agent}
                          </span>
                          <div className="flex items-center gap-2">
                            <span className="text-slate-500 text-[11px]">
                              {agent.avg_latency_ms != null ? `${agent.avg_latency_ms} ms` : "N/A"}
                            </span>
                            <span className={`px-2 py-0.5 font-extrabold text-[10px] rounded-md ${
                              agent.success_rate != null
                                ? "bg-emerald-100 text-emerald-800"
                                : "bg-slate-200 text-slate-600"
                            }`}>
                              {agent.success_rate != null ? `${agent.success_rate}%` : "Awaiting Data"}
                            </span>
                          </div>
                        </div>

                        {/* Latency meter */}
                        <div className="w-full bg-slate-200 rounded-full h-2 overflow-hidden">
                          <div
                            className="bg-violet-600 h-2 rounded-full transition-all duration-500"
                            style={{ width: `${agent.avg_latency_ms != null ? Math.min((agent.avg_latency_ms / 300) * 100, 100) : 0}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* GRAPH 7: ARTICLE DEPTH & WORD LENGTH HISTOGRAM */}
              {data.word_length && (
                <div className="bg-white p-6 rounded-2xl shadow-xs border border-slate-200/80 space-y-6 flex flex-col justify-between">
                  <div>
                    <div className="flex items-center justify-between border-b border-slate-100 pb-3">
                      <h2 className="text-base font-bold text-slate-900 flex items-center gap-2">
                        <FileText className="w-5 h-5 text-emerald-600" />
                        Article Depth & Word Count Distribution
                      </h2>
                      <span className="text-xs text-slate-400 font-semibold">Structure Analysis</span>
                    </div>
                    <p className="text-xs text-slate-500 mt-2">
                      Categorization of stored articles by structural depth and word count length.
                    </p>
                  </div>

                  <div className="space-y-4">
                    {data.word_length.map((w, idx) => {
                      const colors = ["bg-sky-500", "bg-indigo-500", "bg-violet-500", "bg-purple-600"];
                      const color = colors[idx % colors.length];

                      return (
                        <div key={w.range} className="space-y-1.5">
                          <div className="flex justify-between text-xs font-semibold">
                            <span className="text-slate-800">{w.range}</span>
                            <span className="text-slate-500">
                              <span className="font-bold text-slate-800">{w.percentage}%</span>{" "}
                              <span className="text-slate-400 font-normal">({w.count} articles)</span>
                            </span>
                          </div>
                          <div className="w-full bg-slate-100 rounded-full h-3 overflow-hidden">
                            <div
                              className={`${color} h-3 rounded-full transition-all duration-500`}
                              style={{ width: `${Math.max(w.percentage, 3)}%` }}
                            />
                          </div>
                        </div>
                      );
                    })}
                  </div>

                  <div className="p-4 bg-slate-900 text-white rounded-xl flex items-center justify-between">
                    <div className="space-y-0.5">
                      <span className="text-[10px] font-bold text-indigo-400 uppercase tracking-wider block">Average Depth</span>
                      <span className="text-xs font-medium text-slate-300">Corpus is balanced between quick briefs & detailed coverage</span>
                    </div>
                    <Sparkles className="w-5 h-5 text-amber-400" />
                  </div>
                </div>
              )}
            </div>
          )}

          {/* DEDUPLICATION & ARCHITECTURAL SUMMARY BANNER */}
          <div className="bg-gradient-to-r from-slate-900 via-indigo-950 to-slate-900 text-white p-6 md:p-8 rounded-2xl shadow-xl flex flex-col md:flex-row items-center justify-between gap-6 border border-slate-800">
            <div className="space-y-1.5 max-w-xl">
              <div className="flex items-center gap-2">
                <span className="text-xs font-extrabold text-indigo-400 uppercase tracking-widest">MCP Architecture Efficiency</span>
                <span className="px-2 py-0.5 rounded bg-indigo-500/20 text-indigo-300 text-[10px] font-bold border border-indigo-500/30">
                  FastMCP 2.0
                </span>
              </div>
              <h3 className="text-xl font-extrabold text-white">Live Real-time News Ingestion & Deduplication</h3>
              <p className="text-slate-300 text-xs leading-relaxed">
                Raw RSS feeds parsed by BeautifulSoup, sanitized, vector-embedded with BGE-M3, and deduplicated in memory.
                Prevented <strong>{data.duplicates_avoided || 0} duplicate articles</strong> from clogging storage and search indices.
              </p>
            </div>

            <div className="flex items-center gap-4 w-full md:w-auto justify-end">
              <div className="text-center bg-slate-800/90 px-6 py-4 rounded-xl border border-slate-700/80 shadow-inner">
                <span className="text-3xl font-black text-white block">{data.total_articles || 0}</span>
                <span className="text-[10px] text-slate-400 uppercase font-bold tracking-wider">Clean Articles</span>
              </div>
              <div className="text-center bg-amber-500/10 px-6 py-4 rounded-xl border border-amber-500/30 shadow-inner">
                <span className="text-3xl font-black text-amber-400 block">{data.duplicates_avoided || 0}</span>
                <span className="text-[10px] text-amber-300 uppercase font-bold tracking-wider">Duplicates Dropped</span>
              </div>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}
