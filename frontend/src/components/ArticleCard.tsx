import { ExternalLink } from "lucide-react";

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

export interface ArticleCardProps {
  article: Article;
  accentColor?: "indigo" | "amber" | "emerald";
}

export default function ArticleCard({ article, accentColor = "indigo" }: ArticleCardProps) {
  const colorConfig = {
    indigo: {
      tag: "bg-indigo-50 text-indigo-700 border-indigo-100",
      hoverBorder: "hover:border-indigo-300",
      link: "text-indigo-600 hover:text-indigo-800",
      btn: "bg-indigo-600 hover:bg-indigo-700 border-indigo-500"
    },
    amber: {
      tag: "bg-amber-50 text-amber-800 border-amber-100",
      hoverBorder: "hover:border-amber-300",
      link: "text-amber-600 hover:text-amber-800",
      btn: "bg-amber-600 hover:bg-amber-700 border-amber-500"
    },
    emerald: {
      tag: "bg-emerald-50 text-emerald-800 border-emerald-100",
      hoverBorder: "hover:border-emerald-300",
      link: "text-emerald-600 hover:text-emerald-800",
      btn: "bg-emerald-600 hover:bg-emerald-700 border-emerald-500"
    }
  }[accentColor];

  const targetUrl = article.url && article.url !== "#" 
    ? article.url 
    : `https://news.google.com/search?q=${encodeURIComponent(article.title)}`;

  return (
    <div className={`p-4 bg-white rounded-2xl border border-slate-200/80 shadow-2xs hover:shadow-md ${colorConfig.hoverBorder} transition-all flex flex-col md:flex-row items-start md:items-center justify-between gap-4`}>
      <div className="space-y-1.5 max-w-2xl flex-1">
        <div className="flex items-center gap-2 flex-wrap">
          {article.category && (
            <span className={`px-2.5 py-0.5 text-[10px] font-extrabold rounded-md uppercase tracking-wider border ${colorConfig.tag}`}>
              {article.category}
            </span>
          )}
          <span className="text-xs font-extrabold text-slate-700">{article.source}</span>
          {article.published_date && (
            <span className="text-[11px] text-slate-400">• {new Date(article.published_date).toLocaleDateString()}</span>
          )}
        </div>

        <a
          href={targetUrl}
          target="_blank"
          rel="noopener noreferrer"
          className="text-sm font-extrabold text-slate-900 hover:text-indigo-600 transition-colors leading-snug block"
        >
          {article.title}
        </a>

        {/* Visible URL string */}
        {targetUrl && (
          <a
            href={targetUrl}
            target="_blank"
            rel="noopener noreferrer"
            className={`text-[11px] font-mono ${colorConfig.link} hover:underline flex items-center gap-1 truncate max-w-xl`}
          >
            <ExternalLink className="w-3 h-3 shrink-0" />
            {targetUrl}
          </a>
        )}

        {(article.cleaned_content || article.content || article.summary) && (
          <p className="text-xs text-slate-500 line-clamp-2 leading-relaxed">
            {article.cleaned_content || article.content || article.summary}
          </p>
        )}
      </div>

      <a
        href={targetUrl}
        target="_blank"
        rel="noopener noreferrer"
        className={`px-4 py-2 ${colorConfig.btn} text-white text-xs font-extrabold rounded-xl transition-all flex items-center gap-1.5 shrink-0 shadow-sm hover:shadow-md cursor-pointer active:scale-95 text-center`}
      >
        Read Web Article <ExternalLink className="w-3.5 h-3.5" />
      </a>
    </div>
  );
}
