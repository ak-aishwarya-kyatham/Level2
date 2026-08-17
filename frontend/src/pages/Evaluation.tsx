import { useState, useEffect } from "react";
import { 
  ShieldCheck, 
  Target, 
  CheckCircle2, 
  Gauge, 
  Award, 
  RefreshCw,
  Filter,
  CheckCheck
} from "lucide-react";
import apiClient from "../api/client";

interface EvaluationMetric {
  name: string;
  metric_key: string;
  score: number;
  percentage: number;
  target: string;
  status: string;
  category?: string;
  value_text?: string;
  description: string;
}

interface AnalyticsData {
  evaluation_metrics?: EvaluationMetric[];
  total_articles?: number;
  duplicates_avoided?: number;
}

export default function Evaluation() {
  const [data, setData] = useState<AnalyticsData | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [selectedCategory, setSelectedCategory] = useState<string>("All");

  const fetchMetrics = async () => {
    setLoading(true);
    try {
      const res = await apiClient.get("/api/analytics/metrics");
      setData(res.data);
    } catch (err) {
      console.error("Error fetching evaluation metrics:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchMetrics();
    window.addEventListener("liveNewsRefreshed", fetchMetrics);
    return () => window.removeEventListener("liveNewsRefreshed", fetchMetrics);
  }, []);

  const runEvaluationBenchmark = async () => {
    setLoading(true);
    try {
      await apiClient.post("/api/analytics/evaluate");
      await fetchMetrics();
    } catch (err) {
      console.error("Error triggering evaluation benchmark:", err);
      await fetchMetrics();
    } finally {
      setLoading(false);
    }
  };

  // Evaluation Metrics Suite derived strictly from backend API
  const evaluationList: EvaluationMetric[] = data?.evaluation_metrics || [];

  const categoriesList = ["All", "Retrieval Quality", "RAG Integrity", "Workflow Triage", "Ingestion Hygiene", "Classification", "System SLA"];

  const filteredMetrics = selectedCategory === "All"
    ? evaluationList
    : evaluationList.filter((m) => m.category === selectedCategory || (selectedCategory === "System SLA" && m.category === "Memory Caching"));

  // Dynamic Composite score calculation from non-null evaluation metrics
  const validPercentages = evaluationList
    .map((m) => m.percentage)
    .filter((p): p is number => p != null && !isNaN(p));

  const totalScore = validPercentages.length > 0
    ? Math.round(validPercentages.reduce((a, b) => a + b, 0) / validPercentages.length)
    : null;

  const passedCount = evaluationList.filter((m) => m.status === "PASSED").length;
  const totalCount = evaluationList.length;

  return (
    <div className="space-y-8 max-w-6xl mx-auto pb-12">
      {/* Header */}
      <div className="bg-white p-6 md:p-8 rounded-2xl shadow-xs border border-slate-200/80 flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div className="flex items-center gap-3.5">
          <div className="p-3 bg-emerald-600 text-white rounded-xl shadow-md shadow-emerald-200">
            <ShieldCheck className="w-6 h-6" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-2xl font-extrabold text-slate-900 tracking-tight">
                System Evaluation & Benchmark Suite
              </h1>
              <span className={`inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[10px] font-bold ${
                passedCount > 0
                  ? "bg-emerald-100 text-emerald-800 border border-emerald-200"
                  : "bg-amber-100 text-amber-800 border border-amber-200"
              }`}>
                <CheckCircle2 className="w-3 h-3 text-emerald-600" />
                {totalCount > 0 ? `${passedCount}/${totalCount} METRICS PASSED` : "AWAITING EVALUATION"}
              </span>
            </div>
            <p className="text-xs text-slate-500 mt-1">
              Dynamic evaluation suite measuring RAG precision, groundedness, routing accuracy, latency, and classification metrics from actual execution runs.
            </p>
          </div>
        </div>

        <button
          onClick={runEvaluationBenchmark}
          disabled={loading}
          className="px-4 py-2.5 bg-emerald-50 hover:bg-emerald-100 text-emerald-700 text-xs font-bold rounded-xl transition-all duration-200 flex items-center gap-2 cursor-pointer border border-emerald-100 shadow-2xs active:scale-95"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin" : ""}`} />
          Run Dynamic Benchmark
        </button>
      </div>

      {/* Main Overall Composite Score Card */}
      <div className="bg-gradient-to-r from-slate-900 via-indigo-950 to-slate-900 text-white p-6 md:p-8 rounded-2xl shadow-xl border border-slate-800 flex flex-col md:flex-row items-center justify-between gap-6">
        <div className="space-y-2 max-w-xl">
          <div className="flex items-center gap-2">
            <Award className="w-5 h-5 text-amber-400" />
            <span className="text-xs font-extrabold text-indigo-300 uppercase tracking-widest">Composite System Health Score</span>
            <span className={`px-2.5 py-0.5 rounded text-[10px] font-extrabold border ${
              totalScore !== null
                ? "bg-emerald-500/20 text-emerald-300 border-emerald-500/30"
                : "bg-slate-700 text-slate-300 border-slate-600"
            }`}>
              {totalScore !== null ? `${totalScore}% DYNAMIC SCORE` : "AWAITING RUN"}
            </span>
          </div>
          <h2 className="text-2xl font-extrabold text-white">Dynamic AI & RAG Evaluation Metrics</h2>
          <p className="text-slate-300 text-xs leading-relaxed">
            Evaluated against actual pipeline executions and ground truth benchmarks in <code>backend/app/utils/evaluator.py</code>.
            Metrics reflect true measured performance across active queries.
          </p>
        </div>

        <div className="flex items-center gap-4">
          <div className="text-center bg-slate-800/90 px-6 py-4 rounded-2xl border border-slate-700/80 shadow-inner">
            <span className="text-3xl font-black text-emerald-400 block">
              {totalScore !== null ? `${totalScore}%` : "N/A"}
            </span>
            <span className="text-[10px] text-slate-400 uppercase font-bold tracking-wider">Quality Score</span>
          </div>
          <div className="text-center bg-emerald-500/10 px-6 py-4 rounded-2xl border border-emerald-500/30 shadow-inner">
            <span className="text-3xl font-black text-white block">
              {totalCount > 0 ? `${passedCount} / ${totalCount}` : "N/A"}
            </span>
            <span className="text-[10px] text-emerald-300 uppercase font-bold tracking-wider">Metrics Passed</span>
          </div>
        </div>
      </div>

      {/* Category Filter Pills */}
      <div className="flex items-center gap-2 overflow-x-auto pb-1">
        <span className="text-xs font-bold text-slate-400 uppercase tracking-wider flex items-center gap-1 mr-2">
          <Filter className="w-3.5 h-3.5" /> Filter Metrics:
        </span>
        {categoriesList.map((cat) => (
          <button
            key={cat}
            onClick={() => setSelectedCategory(cat)}
            className={`px-3 py-1.5 rounded-xl text-xs font-semibold transition-all cursor-pointer ${
              selectedCategory === cat
                ? "bg-emerald-600 text-white shadow-xs"
                : "bg-white text-slate-600 hover:bg-slate-100 border border-slate-200"
            }`}
          >
            {cat}
          </button>
        ))}
      </div>

      {loading ? (
        <div className="bg-white p-16 rounded-2xl text-center shadow-xs border border-slate-200/80 space-y-3">
          <RefreshCw className="w-8 h-8 text-emerald-600 animate-spin mx-auto" />
          <p className="text-xs font-semibold text-slate-600">Running dynamic evaluation benchmark across query suite...</p>
        </div>
      ) : (
        /* Evaluation Metrics Cards Grid */
        <div className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {filteredMetrics.map((m: any) => {
              const displayVal = m.score != null ? (m.value_text || `${m.percentage}%`) : "N/A";
              const isUnavailable = m.status === "UNAVAILABLE" || m.score == null;

              return (
                <div 
                  key={m.metric_key} 
                  className="bg-white p-6 rounded-2xl border border-slate-200/80 shadow-xs hover:shadow-md hover:border-emerald-200 transition-all space-y-4 flex flex-col justify-between"
                >
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-bold text-slate-800 flex items-center gap-2">
                        <div className="p-2 bg-indigo-50 text-indigo-600 rounded-lg">
                          <Target className="w-4 h-4" />
                        </div>
                        {m.name}
                      </span>
                      <span className={`px-2 py-0.5 text-[10px] font-extrabold rounded-md border ${
                        isUnavailable
                          ? "bg-slate-100 text-slate-600 border-slate-200"
                          : m.status === "PASSED"
                          ? "bg-emerald-100 text-emerald-800 border-emerald-200"
                          : "bg-rose-100 text-rose-800 border-rose-200"
                      }`}>
                        {m.status}
                      </span>
                    </div>

                    {m.category && (
                      <span className="inline-block text-[10px] font-bold text-indigo-600 bg-indigo-50 px-2 py-0.5 rounded-md border border-indigo-100">
                        {m.category}
                      </span>
                    )}

                    <div className="flex items-baseline justify-between pt-1">
                      <span className="text-3xl font-black text-slate-900">{displayVal}</span>
                      <span className="text-xs font-bold text-slate-500 bg-slate-100 px-2 py-1 rounded-md">
                        Target: {m.target}
                      </span>
                    </div>

                    {/* Progress Bar */}
                    <div className="w-full bg-slate-100 rounded-full h-2.5 overflow-hidden">
                      <div
                        className={`h-2.5 rounded-full transition-all duration-500 ${
                          isUnavailable ? "bg-slate-300" : "bg-emerald-600"
                        }`}
                        style={{ width: `${m.percentage != null ? Math.min(m.percentage, 100) : 0}%` }}
                      />
                    </div>

                    <p className="text-xs text-slate-500 leading-relaxed pt-1">
                      {m.description}
                    </p>

                    {m.source && (
                      <div className="text-[10px] text-slate-400 bg-slate-50 p-2 rounded-lg border border-slate-100 space-y-0.5">
                        <div><span className="font-semibold text-slate-600">Source:</span> {m.source}</div>
                        {m.calculation && <div><span className="font-semibold text-slate-600">Calculation:</span> {m.calculation}</div>}
                      </div>
                    )}
                  </div>

                  <div className="pt-3 border-t border-slate-100 flex items-center justify-between text-[11px] text-slate-400">
                    <span className="font-mono text-[10px] text-slate-400">{m.metric_key}</span>
                    <span className={`font-bold flex items-center gap-1 ${isUnavailable ? "text-slate-400" : "text-emerald-600"}`}>
                      <CheckCheck className="w-3.5 h-3.5" /> {isUnavailable ? "Awaiting Data" : "Dynamic Data"}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>

          {/* Test Architecture Banner */}
          <div className="bg-white p-6 rounded-2xl border border-slate-200/80 shadow-xs flex flex-col md:flex-row items-center justify-between gap-6">
            <div className="flex items-center gap-4">
              <div className="p-3.5 bg-slate-100 text-slate-700 rounded-2xl">
                <Gauge className="w-6 h-6 text-indigo-600" />
              </div>
              <div>
                <h3 className="text-base font-extrabold text-slate-900">Pytest Automated Continuous Evaluation</h3>
                <p className="text-xs text-slate-500 mt-0.5">
                  All 10 evaluation benchmarks are executed via <code>backend/tests/test_evaluation.py</code> on every build.
                </p>
              </div>
            </div>

            <div className="flex items-center gap-3">
              <div className="px-4 py-2 bg-slate-100 text-slate-700 font-mono text-xs rounded-xl border border-slate-200">
                python -c "import tests.test_evaluation"
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
