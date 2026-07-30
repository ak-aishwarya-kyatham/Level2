import { 
  X, 
  ExternalLink, 
  RefreshCw, 
  Newspaper, 
  MessageSquareText
} from "lucide-react";


export interface Article {
  id?: string;
  title: string;
  source: string;
  url: string;
  category?: string;
  published_date: string;
  content?: string;
  cleaned_content?: string;
  summary?: string;
}

export interface ArticleModalProps {
  isOpen: boolean;
  onClose: () => void;
  title: string;
  subtitle?: string;
  category?: string;
  badgeText?: string;
  description?: string;
  primaryUrl?: string;
  articles: Article[];
  loading?: boolean;
  accentColor?: "amber" | "indigo" | "emerald";
  onAskAI?: (prompt: string) => void;
}

export default function ArticleModal({
  isOpen,
  onClose,
  title,
  subtitle,
  category,
  badgeText,
  description,
  primaryUrl,
  articles,
  loading = false,
  accentColor = "amber",
  onAskAI
}: ArticleModalProps) {
  if (!isOpen) return null;

  const colorClasses = {
    amber: {
      headerBg: "bg-gradient-to-r from-amber-50 via-orange-50 to-amber-50",
      badgeBg: "bg-amber-600 text-white",
      tagBg: "bg-amber-100 text-amber-800",
      borderHover: "hover:border-amber-300",
      btnBg: "bg-amber-600 hover:bg-amber-700 border-amber-500",
      iconColor: "text-amber-600",
      spinnerColor: "text-amber-500",
      overviewBg: "bg-amber-50/70 border-amber-200/80"
    },
    indigo: {
      headerBg: "bg-gradient-to-r from-indigo-50 via-slate-50 to-indigo-50",
      badgeBg: "bg-indigo-600 text-white",
      tagBg: "bg-indigo-100 text-indigo-800",
      borderHover: "hover:border-indigo-300",
      btnBg: "bg-indigo-600 hover:bg-indigo-700 border-indigo-500",
      iconColor: "text-indigo-600",
      spinnerColor: "text-indigo-500",
      overviewBg: "bg-indigo-50/70 border-indigo-200/80"
    },
    emerald: {
      headerBg: "bg-gradient-to-r from-emerald-50 via-teal-50 to-emerald-50",
      badgeBg: "bg-emerald-600 text-white",
      tagBg: "bg-emerald-100 text-emerald-800",
      borderHover: "hover:border-emerald-300",
      btnBg: "bg-emerald-600 hover:bg-emerald-700 border-emerald-500",
      iconColor: "text-emerald-600",
      spinnerColor: "text-emerald-500",
      overviewBg: "bg-emerald-50/70 border-emerald-200/80"
    }
  }[accentColor];

  const mainArticleUrl = primaryUrl || articles[0]?.url || `https://news.google.com/search?q=${encodeURIComponent(title)}`;

  return (
    <div className="fixed inset-0 z-50 bg-slate-900/60 backdrop-blur-xs flex items-center justify-center p-4">
      <div 
        className="bg-white rounded-3xl max-w-3xl w-full max-h-[85vh] flex flex-col shadow-2xl border border-slate-200 animate-in zoom-in-95 duration-200"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Modal Header */}
        <div className={`p-6 border-b border-slate-100 flex items-center justify-between ${colorClasses.headerBg} rounded-t-3xl`}>
          <div className="space-y-1 max-w-xl">
            <div className="flex items-center gap-2 flex-wrap">
              {category && (
                <span className={`px-2.5 py-0.5 ${colorClasses.badgeBg} text-[10px] font-extrabold rounded-md uppercase tracking-wider`}>
                  {category}
                </span>
              )}
              {badgeText && (
                <span className="text-xs font-bold text-emerald-700 bg-emerald-100 px-2 py-0.5 rounded-md">
                  {badgeText}
                </span>
              )}
            </div>
            <h3 className="text-xl font-extrabold text-slate-900 leading-snug">
              {title}
            </h3>
            {subtitle && (
              <p className="text-xs text-slate-500">{subtitle}</p>
            )}
          </div>
          <div className="flex items-center gap-3">
            <a
              href={mainArticleUrl}
              target="_blank"
              rel="noopener noreferrer"
              className={`px-3.5 py-2 ${colorClasses.btnBg} text-white text-xs font-extrabold rounded-xl transition-all flex items-center gap-1.5 shrink-0 shadow-md hover:shadow-lg cursor-pointer active:scale-95`}
              title="Open Article Webpage"
            >
              Open Web Link <ExternalLink className="w-3.5 h-3.5" />
            </a>
            <button 
              onClick={onClose}
              className="p-2 hover:bg-white text-slate-400 hover:text-slate-700 rounded-full transition-all cursor-pointer"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Modal Body */}
        <div className="p-6 overflow-y-auto space-y-5 flex-1">
          {/* Main Article Link Banner */}
          <div className="p-3 bg-slate-100 rounded-2xl border border-slate-200 flex flex-col sm:flex-row sm:items-center justify-between gap-2 text-xs">
            <div className="flex items-center gap-2 truncate">
              <span className="font-extrabold text-slate-700 shrink-0">Direct Link:</span>
              <a
                href={mainArticleUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="text-indigo-600 hover:text-indigo-800 hover:underline font-mono truncate max-w-md"
              >
                {mainArticleUrl}
              </a>
            </div>
            <a
              href={mainArticleUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="text-indigo-700 font-extrabold hover:underline flex items-center gap-1 shrink-0 self-start sm:self-auto"
            >
              Visit Publisher Page <ExternalLink className="w-3.5 h-3.5" />
            </a>
          </div>

          {description && (
            <div className={`p-4 rounded-2xl border text-xs text-slate-600 space-y-2 ${colorClasses.overviewBg}`}>
              <p className="font-semibold text-slate-800">
                Topic Overview: {description}
              </p>
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 text-slate-600 pt-2 border-t border-slate-200/60">
                <span>Coverage Volume: <strong>{articles.length} Verified Article{articles.length !== 1 ? "s" : ""}</strong></span>
                {onAskAI && (
                  <button 
                    onClick={() => {
                      onClose();
                      onAskAI(title);
                    }}
                    className="text-indigo-700 font-extrabold hover:underline flex items-center gap-1 cursor-pointer bg-white px-3 py-1 rounded-lg border border-indigo-200 shadow-2xs self-start sm:self-auto"
                  >
                    <MessageSquareText className="w-3.5 h-3.5 text-indigo-600" /> Ask AI Agent About This Topic
                  </button>
                )}
              </div>
            </div>
          )}

          <div className="flex items-center justify-between pt-1">
            <h4 className="text-xs font-extrabold text-slate-700 uppercase tracking-wider flex items-center gap-1.5">
              <Newspaper className={`w-4 h-4 ${colorClasses.iconColor}`} />
              Articles in Cluster ({articles.length}):
            </h4>
          </div>

          {loading ? (
            <div className="p-12 text-center text-xs font-bold text-slate-400 space-y-2">
              <RefreshCw className={`w-6 h-6 animate-spin mx-auto ${colorClasses.spinnerColor}`} />
              <p>Fetching real-time articles for cluster...</p>
            </div>
          ) : articles.length === 0 ? (
            <div className="p-6 bg-slate-50 rounded-2xl border border-slate-200 space-y-4">
              <div className="space-y-1">
                <h5 className="text-sm font-bold text-slate-900">{title}</h5>
                <p className="text-xs text-slate-500">Live article story page</p>
                <a
                  href={mainArticleUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-xs text-indigo-600 font-mono hover:underline block truncate"
                >
                  {mainArticleUrl}
                </a>
              </div>
              <a
                href={mainArticleUrl}
                target="_blank"
                rel="noopener noreferrer"
                className={`px-4 py-2.5 ${colorClasses.btnBg} text-white text-xs font-extrabold rounded-xl inline-flex items-center gap-2 shadow-md`}
              >
                Read Article on Web <ExternalLink className="w-4 h-4" />
              </a>
            </div>
          ) : (
            <div className="space-y-3">
              {articles.map((art, idx) => (
                <div 
                  key={art.id || idx} 
                  className={`p-4 bg-slate-50/80 rounded-2xl border border-slate-200/90 hover:bg-white ${colorClasses.borderHover} hover:shadow-md transition-all flex flex-col md:flex-row items-start md:items-center justify-between gap-4`}
                >
                  <div className="space-y-1.5 max-w-xl">
                    <div className="flex items-center gap-2 flex-wrap">
                      {art.category && (
                        <span className={`px-2 py-0.5 ${colorClasses.tagBg} text-[10px] font-bold rounded-md`}>
                          {art.category}
                        </span>
                      )}
                      <span className="text-xs font-bold text-slate-700">{art.source}</span>
                      {art.published_date && (
                        <span className="text-[10px] text-slate-400">• {new Date(art.published_date).toLocaleDateString()}</span>
                      )}
                    </div>
                    
                    <a 
                      href={art.url || mainArticleUrl}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-sm font-extrabold text-slate-900 hover:text-indigo-600 transition-colors leading-snug block"
                    >
                      {art.title}
                    </a>

                    {/* Visible URL link */}
                    {(art.url || mainArticleUrl) && (
                      <a
                        href={art.url || mainArticleUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-[11px] text-indigo-600 hover:text-indigo-800 hover:underline font-mono flex items-center gap-1 truncate max-w-lg"
                      >
                        <ExternalLink className="w-3 h-3 shrink-0" />
                        {art.url || mainArticleUrl}
                      </a>
                    )}

                    {(art.cleaned_content || art.content || art.summary) && (
                      <p className="text-xs text-slate-500 line-clamp-2 leading-relaxed">
                        {art.cleaned_content || art.content || art.summary}
                      </p>
                    )}
                  </div>

                  <a
                    href={art.url || mainArticleUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    className={`px-4 py-2 ${colorClasses.btnBg} text-white text-xs font-extrabold rounded-xl transition-all flex items-center gap-1.5 shrink-0 shadow-md hover:shadow-lg cursor-pointer active:scale-95 text-center`}
                  >
                    Read Original Article <ExternalLink className="w-4 h-4" />
                  </a>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Modal Footer */}
        <div className="p-4 border-t border-slate-100 bg-slate-50 rounded-b-3xl flex items-center justify-between text-xs text-slate-500">
          <span>Live News Intel Stream</span>
          <div className="flex items-center gap-3">
            <a
              href={mainArticleUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="text-indigo-600 font-extrabold hover:underline flex items-center gap-1"
            >
              Open Web URL <ExternalLink className="w-3.5 h-3.5" />
            </a>
            <button 
              onClick={onClose}
              className="px-5 py-2 bg-white hover:bg-slate-200 text-slate-700 font-extrabold rounded-xl border border-slate-200 cursor-pointer shadow-2xs"
            >
              Close Window
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}




