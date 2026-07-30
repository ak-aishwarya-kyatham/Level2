import type { LucideIcon } from "lucide-react";


export interface MetricCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  icon: LucideIcon;
  iconBgColor?: string;
  iconTextColor?: string;
  badgeText?: string;
  badgeColor?: "emerald" | "indigo" | "amber" | "slate";
}

export default function MetricCard({
  title,
  value,
  subtitle,
  icon: Icon,
  iconBgColor = "bg-indigo-50",
  iconTextColor = "text-indigo-600",
  badgeText,
  badgeColor = "emerald"
}: MetricCardProps) {
  const badgeClasses = {
    emerald: "bg-emerald-50 text-emerald-700 border-emerald-100",
    indigo: "bg-indigo-50 text-indigo-700 border-indigo-100",
    amber: "bg-amber-50 text-amber-700 border-amber-100",
    slate: "bg-slate-100 text-slate-700 border-slate-200"
  }[badgeColor];

  return (
    <div className="bg-white p-6 rounded-2xl shadow-xs border border-slate-200/80 hover:shadow-md transition-all flex items-start justify-between">
      <div className="space-y-2">
        <span className="text-xs font-semibold text-slate-500 block uppercase tracking-wider">
          {title}
        </span>
        <h3 className="text-2xl font-black text-slate-900 tracking-tight">
          {value}
        </h3>
        {subtitle && (
          <p className="text-xs text-slate-500 leading-snug">{subtitle}</p>
        )}
      </div>

      <div className="flex flex-col items-end gap-2">
        <div className={`p-3 rounded-xl ${iconBgColor} ${iconTextColor} border border-slate-100/60 shadow-2xs`}>
          <Icon className="w-6 h-6" />
        </div>
        {badgeText && (
          <span className={`text-[10px] font-bold px-2 py-0.5 rounded-md border ${badgeClasses}`}>
            {badgeText}
          </span>
        )}
      </div>
    </div>
  );
}
