# Software Requirements Specification (SRS)
## NewsIntel AI — Multi-Agent Enterprise News Intelligence Platform

**Document Version:** 2.0.0  
**Status:** Approved / Active  
**Author:** DeepMind AI Pair Programming / Aishwarya Kyatham  
**Standard:** IEEE 830 / L2 Capstone Specification  

---

### 1. Introduction

#### 1.1 Purpose
The purpose of this document is to specify the complete functional, non-functional, and architectural requirements for **NewsIntel AI**, a Level 2 (L2) tool-using autonomous agent platform. NewsIntel AI collects multi-source RSS news feeds, extracts structured intelligence, exposes news operations via Model Context Protocol (MCP) servers, and executes a multi-turn ReAcT (Reasoning + Acting) decision loop powered by local large language models (e.g. Qwen 2.5) with observation-grounded reflection.

#### 1.2 Scope
NewsIntel AI automates the news consumption, analysis, and verification pipeline for enterprise users, intelligence analysts, and researchers. The system supports:
- Automated multi-source RSS ingestion and vector indexing.
- Dynamic tool discovery and standard I/O execution via Model Context Protocol (FastMCP).
- Multi-step goal-directed agent reasoning with stateful observation memory.
- Factual grounding and reflection verifiers to eliminate hallucinations.
- Real-time quantitative evaluation of routing accuracy, latency, and groundedness.

#### 1.3 Definitions and Acronyms
- **MCP**: Model Context Protocol — open standard protocol developed by Anthropic for exposing tools and resources to LLMs over standard JSON-RPC.
- **ReAcT**: Reasoning and Acting loop (`Decide -> Act -> Observe -> Reflect`).
- **LangGraph**: Stateful graph orchestration framework for cyclic agent workflows.
- **Qdrant**: High-performance vector database for semantic similarity and duplicate detection.
- **Precision@K / MRR**: Retrieval quality metrics for ranked search results.

---

### 2. Overall Description

#### 2.1 Product Perspective
NewsIntel AI operates as a self-contained, modular full-stack application:
- **Backend API**: FastAPI (Python 3.11+) exposing REST endpoints for chat, analytics, sources, and comparison.
- **MCP Server**: FastMCP server running on `stdio` transport, encapsulating 5 typed tools and 2 resources.
- **MCP Client**: Standard `ClientSession` manager with connection pooling and subprocess lifecycle management.
- **Agent Workflow**: LangGraph cyclic state machine enforcing maximum iteration caps (5 iterations), Pydantic action schema validation, and reflection guardrails.
- **Local Inference**: Local Ollama runtime hosting `qwen2.5:3b` with deterministic extractive fallback.
- **Frontend Dashboard**: React + TypeScript + Vite interactive intelligence workspace.

#### 2.2 User Personas
1. **Intelligence Analyst**: Queries multi-source news on specific topics (e.g., "AI developments", "Semiconductor supply chain"), requests source comparisons, and expects grounded, cited executive briefings.
2. **System Evaluator / Auditor**: Inspects dynamic evaluation metrics, verifies unmanipulated latency measurements, and confirms genuine MCP protocol boundaries.
3. **News Curator**: Monitors live feed health, duplicate prevention metrics, and category distributions.

---

### 3. Functional Requirements

#### 3.1 Ingestion & Normalization Engine (REQ-INGEST)
- **REQ-INGEST-01**: The system shall ingest live RSS feeds from at least 8 major news sources (The Hindu, Indian Express, NDTV, TechCrunch, BBC, VentureBeat, Times of India, Hindustan Times).
- **REQ-INGEST-02**: Ingested articles shall be sanitized (HTML tag removal, whitespace normalization, script injection prevention).
- **REQ-INGEST-03**: Articles shall be categorized into defined domains (*Technology, Business, Politics, Sports, General News, Entertainment*).
- **REQ-INGEST-04**: Duplicate and near-duplicate articles shall be identified and filtered using embedding cosine similarity (>0.88 threshold).

#### 3.2 Model Context Protocol (MCP) Subsystem (REQ-MCP)
- **REQ-MCP-01**: The MCP server (`mcp_server.py`) shall run independently over `stdio` transport.
- **REQ-MCP-02**: The server shall register and expose 5 typed tools:
  1. `search_live_news(query, category, source, limit)`
  2. `fetch_latest_rss_feeds()`
  3. `get_dashboard_analytics()`
  4. `compare_news_sources(source1, source2)`
  5. `get_articles_by_category(category, limit)`
- **REQ-MCP-03**: The server shall expose 2 URI-accessible resources:
  1. `news://store/articles`: Complete stored article dataset.
  2. `news://analytics/metrics`: Real-time system analytics and distribution metrics.
- **REQ-MCP-04**: The MCP client (`mcp_client.py`) shall establish an asynchronous `ClientSession` with the server subprocess, perform dynamic tool discovery via `list_tools()`, and dispatch execution via `call_tool()`.

#### 3.3 Dynamic ReAcT Agent Loop (REQ-AGENT)
- **REQ-AGENT-01**: The Policy Agent (`policy_agent.py`) shall generate structured JSON conforming to `PolicyAction`:
  - `action`: Strict literal (`"tool"` or `"finish"`).
  - `tool`: String matching a discovered MCP tool name (when `action == "tool"`).
  - `arguments`: Dictionary conforming to the MCP tool's `inputSchema`.
  - `answer`: Grounded response summary (when `action == "finish"`).
  - `thought`: Natural-language reasoning behind the action.
- **REQ-AGENT-02**: The system shall validate LLM outputs against both Pydantic schemas and discovered MCP tool schemas. Invalid outputs shall be returned to the observation history as structured repair feedback.
- **REQ-AGENT-03**: The workflow state (`AgentState`) shall preserve iteration count, full observation history, execution timings, and agent trace.
- **REQ-AGENT-04**: The loop shall enforce a strict upper bound of `MAX_ITERATIONS = 5` to prevent infinite loops.

#### 3.4 Factual Reflection & Grounding (REQ-REFLECT)
- **REQ-REFLECT-01**: Every drafted synthesis shall be evaluated by `ReflectionAgent` comparing claims against collected tool observations.
- **REQ-REFLECT-02**: If unsupported claims are detected, the reflection agent shall return `revise=True` with specific critique, prompting the policy agent to perform targeted search refinement.
- **REQ-REFLECT-03**: The reflection engine shall fail-closed: if the LLM verifier is unreachable, a conservative token-overlap verifier shall run and attach a prominent `[UNVERIFIED]` disclaimer.
- **REQ-REFLECT-04**: All final responses shall include verifiable primary source URLs extracted from tool observations.

#### 3.5 Dynamic Evaluation & Telemetry (REQ-EVAL)
- **REQ-EVAL-01**: Endpoint `POST /api/analytics/evaluate` shall execute the routing benchmark against labeled query datasets without artificial floors or score clamping.
- **REQ-EVAL-02**: Execution latency shall be measured using high-precision monotonic timers (`time.perf_counter()`).
- **REQ-EVAL-03**: Evaluation metrics (Precision@5, MRR@10, Routing Accuracy, Faithfulness, Categorization F1) shall be computed over live and stored dataset runs.

---

### 4. Non-Functional Requirements

#### 4.1 Performance & Latency (NFR-PERF)
- **NFR-PERF-01**: In-memory Redis cache hits shall respond in under 50ms.
- **NFR-PERF-02**: Complete offline pytest test suite execution shall execute cleanly with 0 collection errors and 0 failures.
- **NFR-PERF-03**: MCP tool invocation over stdio transport shall have sub-10ms overhead per call.

#### 4.2 Security & Safety (NFR-SEC)
- **NFR-SEC-01**: Observation sanitization shall redact known prompt injection patterns (`ignore prior instructions`, `override system prompt`).
- **NFR-SEC-02**: JWT secrets shall be securely configured via environment variables with randomized startup defaults.
- **NFR-SEC-03**: Tool argument ranges shall enforce safe bounds (`limit: 1..50`) to prevent denial-of-service or memory exhaustion.

#### 4.3 Reliability & Fault Tolerance (NFR-REL)
- **NFR-REL-01**: The system shall maintain graceful offline degradation: when Ollama is offline, deterministic extractive summarization shall synthesize factual answers from cached articles.
- **NFR-REL-02**: Background tasks shall be tracked and cleanly cancelled upon application shutdown via FastAPI lifespan handlers.

---

### 5. Verification & Compliance Matrix

| Requirement ID | Verification Method | Pass Criteria |
| :--- | :--- | :--- |
| **REQ-MCP-01..04** | Automated Integration Test (`test_mcp_real_session.py`) | Real stdio subprocess initialized, 5 tools discovered, live tool executed. |
| **REQ-AGENT-01..04** | Automated Unit Suite (`test_agentic_loop.py`, `test_policy_action_validation.py`) | All action schema and validation tests pass with 0 errors. |
| **REQ-REFLECT-01..04** | Automated Reflection Suite (`test_reflection_agent.py`) | Factual check, revision routing, and unverified disclaimer verified. |
| **REQ-EVAL-01..03** | Live API Benchmark (`test_analytics_api.py`, `POST /api/analytics/evaluate`) | Unclamped scores returned with real measured latency. |
| **NFR-PERF-02** | Pytest CLI execution (`pytest -q -p no:cacheprovider`) | 89 offline tests pass with 0 collection errors and 0 failures. |
