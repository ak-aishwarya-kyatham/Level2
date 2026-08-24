import { useState, useEffect, useRef } from "react";

import { useLocation } from "react-router-dom";
import { Send, Bot, User, Sparkles, RefreshCw, ExternalLink, Lightbulb } from "lucide-react";
import apiClient from "../api/client";


interface Message {
  role: "user" | "assistant";
  content: string;
  intent?: string;
}

export default function Chat() {
  const location = useLocation();
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const hasHandledPrefill = useRef<string | null>(null);

  const [messages, setMessages] = useState<Message[]>([
    {
      role: "assistant",
      content:
        "Hello! I am NewsIntel AI Assistant. I am connected live to real-time RSS feeds and news outlets. Ask me to summarize recent news, search specific topics, analyze trends, or compare news sources!",
    },
  ]);

  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  // Check if navigate state passed a prefill or query prompt (prevent double execution)
  useEffect(() => {
    if (location.state) {
      const stateObj = location.state as any;
      const prefillMsg = stateObj.prefill || stateObj.query;
      if (prefillMsg && hasHandledPrefill.current !== prefillMsg) {
        hasHandledPrefill.current = prefillMsg;
        window.history.replaceState({}, document.title);
        setInput("");
        handleSend(undefined, prefillMsg);
      }
    }
  }, [location.state]);


  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const handleSend = async (e?: React.FormEvent, customQuery?: string) => {
    if (e) e.preventDefault();
    const queryToSend = customQuery || input;
    if (!queryToSend.trim() || loading) return;

    const userMessage: Message = { role: "user", content: queryToSend };
    setMessages((prev) => [...prev, userMessage]);
    if (!customQuery) setInput("");
    setLoading(true);

    try {
      const response = await apiClient.post(
        "/api/chat/",
        {
          query: queryToSend,
          user_id: "default_user",
        },
        { timeout: 300000 }
      );

      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: response.data.response || "No answer generated.",
          intent: response.data.intent,
        },
      ]);
    } catch (error: any) {
      console.error("Chat error:", error);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content:
            error?.code === "ECONNABORTED"
              ? "⏳ The request timed out. The AI is processing your query — please try again."
              : error?.response
                ? `⚠️ Server error: ${error.response.status} — ${error.response.data?.detail || 'Unknown error'}`
                : "⚠️ Unable to connect to the backend server. Please verify the backend FastAPI process is running on http://localhost:8000.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const samplePrompts = [
    "Summarize top tech news today",
    "What are the global economic updates?",
    "Compare Times of India vs The Hindu coverage",
    "Latest artificial intelligence innovations",
  ];

  // Simple markdown link formatter
  const renderFormattedText = (text: string) => {
    const lines = text.split("\n");
    return lines.map((line, lineIdx) => {
      // Convert markdown links [text](url) to clickable anchor tags
      const linkRegex = /\[([^\]]+)\]\(([^)]+)\)/g;
      const parts = [];
      let lastIndex = 0;
      let match;

      while ((match = linkRegex.exec(line)) !== null) {
        if (match.index > lastIndex) {
          parts.push(line.substring(lastIndex, match.index));
        }
        const linkText = match[1];
        const linkUrl = match[2];
        parts.push(
          <a
            key={`${lineIdx}-${match.index}`}
            href={linkUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="text-indigo-600 hover:text-indigo-800 underline font-semibold inline-flex items-center gap-0.5 mx-1"
          >
            {linkText}
            <ExternalLink className="w-3 h-3 inline" />
          </a>
        );
        lastIndex = linkRegex.lastIndex;
      }

      if (lastIndex < line.length) {
        parts.push(line.substring(lastIndex));
      }

      // Check headers
      if (line.startsWith("### ")) {
        return (
          <h3 key={lineIdx} className="text-base font-bold text-slate-900 mt-3 mb-1">
            {parts}
          </h3>
        );
      }
      if (line.startsWith("**") && line.endsWith("**")) {
        return (
          <p key={lineIdx} className="font-bold text-slate-800 my-1">
            {parts}
          </p>
        );
      }

      return (
        <p key={lineIdx} className="leading-relaxed my-0.5">
          {parts.length > 0 ? parts : line}
        </p>
      );
    });
  };

  return (
    <div className="flex flex-col h-[calc(100vh-7rem)] max-w-5xl mx-auto bg-white rounded-2xl shadow-sm border border-slate-200/80 overflow-hidden">
      {/* Chat Header */}
      <div className="p-4 md:p-5 border-b border-slate-100 bg-slate-900 text-white flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-indigo-600 rounded-xl">
            <Bot className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="text-base font-bold tracking-tight">AI News Intelligence Assistant</h1>
            <p className="text-xs text-indigo-300 flex items-center gap-1.5">
              <Sparkles className="w-3 h-3 text-indigo-400" />
              Grounded in Real-Time RSS & Live News
            </p>
          </div>
        </div>

        <button
          onClick={() =>
            setMessages([
              {
                role: "assistant",
                content:
                  "Chat reset. Ask me anything about current news, trends, or source comparisons!",
              },
            ])
          }
          className="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg text-xs font-medium transition-colors cursor-pointer"
        >
          Clear Chat
        </button>
      </div>

      {/* Messages Thread */}
      <div className="flex-1 overflow-y-auto p-4 md:p-6 space-y-5 bg-slate-50/50">
        {messages.map((msg, i) => (
          <div
            key={i}
            className={`flex gap-3 ${msg.role === "user" ? "justify-end" : "justify-start"}`}
          >
            {msg.role === "assistant" && (
              <div className="w-8 h-8 rounded-xl bg-gradient-to-tr from-indigo-600 to-violet-600 text-white flex items-center justify-center font-bold text-xs shrink-0 shadow-sm mt-1">
                <Bot className="w-4 h-4" />
              </div>
            )}

            <div
              className={`max-w-[85%] sm:max-w-[75%] p-4 rounded-2xl text-xs md:text-sm shadow-xs ${
                msg.role === "user"
                  ? "bg-indigo-600 text-white rounded-br-none"
                  : "bg-white text-slate-800 border border-slate-200/80 rounded-bl-none space-y-1"
              }`}
            >
              {msg.intent && (
                <div className="inline-block px-2 py-0.5 mb-2 rounded bg-indigo-50 text-indigo-700 text-[10px] font-bold uppercase tracking-wider">
                  Intent: {msg.intent}
                </div>
              )}
              <div className="whitespace-pre-wrap leading-relaxed">
                {msg.role === "assistant" ? renderFormattedText(msg.content) : msg.content}
              </div>
            </div>

            {msg.role === "user" && (
              <div className="w-8 h-8 rounded-xl bg-slate-800 text-white flex items-center justify-center font-bold text-xs shrink-0 shadow-sm mt-1">
                <User className="w-4 h-4" />
              </div>
            )}
          </div>
        ))}

        {loading && (
          <div className="flex justify-start gap-3">
            <div className="w-8 h-8 rounded-xl bg-gradient-to-tr from-indigo-600 to-violet-600 text-white flex items-center justify-center font-bold text-xs shrink-0 shadow-sm">
              <Bot className="w-4 h-4" />
            </div>
            <div className="max-w-[75%] p-4 rounded-2xl bg-white border border-slate-200/80 text-slate-700 rounded-bl-none shadow-xs flex items-center gap-3">
              <RefreshCw className="w-4 h-4 text-indigo-600 animate-spin" />
              <span className="text-xs font-semibold text-slate-600">
                Searching live RSS articles & synthesizing response...
              </span>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Prompt Suggestions */}
      <div className="px-4 py-2.5 bg-slate-100/70 border-t border-slate-200/80 flex items-center gap-2 overflow-x-auto scrollbar-none">
        <Lightbulb className="w-3.5 h-3.5 text-amber-500 shrink-0 ml-1" />
        <span className="text-[11px] font-semibold text-slate-500 shrink-0">Try asking:</span>
        {samplePrompts.map((p, idx) => (
          <button
            key={idx}
            onClick={() => handleSend(undefined, p)}
            disabled={loading}
            className="px-3 py-1 bg-white hover:bg-indigo-50 hover:text-indigo-700 text-slate-700 border border-slate-200 rounded-lg text-xs whitespace-nowrap transition-colors cursor-pointer shrink-0 font-medium disabled:opacity-50"
          >
            {p}
          </button>
        ))}
      </div>

      {/* Chat Input Form */}
      <div className="p-4 border-t border-slate-100 bg-white">
        <form onSubmit={(e) => handleSend(e)} className="flex items-center gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            className="flex-1 p-3.5 bg-slate-50 border border-slate-200 rounded-xl text-sm focus:bg-white focus:outline-none focus:ring-2 focus:ring-indigo-500 font-medium transition-all"
            placeholder="Ask about live news, specific events, or trends..."
            disabled={loading}
          />
          <button
            type="submit"
            disabled={loading || !input.trim()}
            className="px-6 py-3.5 bg-indigo-600 hover:bg-indigo-700 disabled:bg-indigo-300 text-white rounded-xl transition-all font-semibold flex items-center gap-2 shadow-sm cursor-pointer"
          >
            <span>Send</span>
            <Send className="w-4 h-4" />
          </button>
        </form>
      </div>
    </div>
  );
}
