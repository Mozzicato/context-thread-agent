# Context Thread Agent

**A context-aware notebook copilot for analytical workflows**

---

## 1. Overview

### Problem

Modern notebooks (SQL, Python, Markdown, charts) accumulate implicit context as they grow:
- Assumptions are unstated
- Reasoning is scattered across cells
- Downstream effects of changes are unclear

Existing AI copilots answer questions out of context, hallucinate, or ignore prior steps. This breaks analytical workflows where reasoning continuity is critical.

### Goal

Build an agent that:
- **Understands** the evolving context of a notebook (not just individual cells)
- **Answers questions only from available context** (no hallucination)
- **Preserves reasoning, assumptions, and dependencies** (maintains audit trail)
- **Cites sources** (enables verification)

This agent augments—not replaces—human analytical thinking. It's a copilot for rigor.

---

## 2. Non-Goals

To keep scope realistic:
- ❌ Not a full notebook IDE
- ❌ Not real-time collaborative editing
- ❌ Not automatic code generation at scale
- ❌ Not model fine-tuning
- ❌ Not a semantic search engine (it's a reasoning engine)

**The focus is context structure, intelligent retrieval, and citation-aware reasoning.**

---

## 3. Core Concept: Context Threads

A **Context Thread** is a structured, evolving representation of notebook state.

Each thread contains:
- **Cells** (code, markdown, SQL) with versioning
- **Outputs** (tables, charts, logs, summaries)
- **Intent metadata** (why this step exists, assumptions)
- **Dependencies** (upstream/downstream relationships)
- **Timeline** (when each step was added or modified)

The agent reasons *within* this thread—never outside it. This ensures consistency and auditability.

---

## 4. User Experience

### Primary Workflow

1. User works in a notebook-like environment
2. Each cell is automatically:
   - Indexed and embedded
   - Tagged with type + inferred intent
   - Linked to its dependencies
3. User asks a contextual question:
   - _"Why did we remove Q4 data?"_
   - _"What's the revenue impact of this filter?"_
4. Agent:
   - Retrieves relevant cells + outputs (semantically)
   - Expands to dependencies (structural)
   - Reconstructs reasoning chain
   - Responds with **cited sources**

### Example Queries

- _"What assumptions does this chart rely on?"_
- _"If I change this filter, what downstream cells break?"_
- _"Where did this metric come from?"_
- _"Summarize the logic so far."_
- _"Show me all transformations that touch the revenue column."_

---

## 5. System Architecture

### High-Level Components

```
Notebook UI
   ↓
Context Extractor (Intent + Structure)
   ↓
Context Index (Vector DB + Dependency Graph)
   ↓
Retrieval Engine (Semantic + Structural)
   ↓
LLM Reasoner (Citation-aware)
   ↓
Response with Sources
```

---

## 6. Context Extraction

Each cell is processed into a **Context Unit**:

```json
{
  "cell_id": "cell_12",
  "type": "python",
  "content": "df = df[df['quarter'] != 'Q4']",
  "output_summary": "Rows reduced from 12,000 → 9,100",
  "intent": "Remove incomplete Q4 data",
  "dependencies": ["cell_10"],
  "timestamp": "2026-01-03T14:32"
}
```

Intent is inferred via:
- Lightweight LLM prompt on cell + outputs
- Optional user override for clarity
- Caching to avoid repeated inference

---

## 7. Indexing Strategy

### Storage
- **Vector DB**: FAISS or Chroma (fast, local)
- **Metadata index**: SQLite or in-memory (dependencies, timestamps)
- **Graph structure**: Adjacency list for dependency traversal

### Embeddings (per cell)
- Cell content
- Output summary
- Inferred intent
- Comments/annotations (if present)

### Permissions (Future)
- User-scoped context threads
- Workspace isolation
- Audit logs

---

## 8. Retrieval Strategy

Retrieval is **multi-stage**:

1. **Semantic search** (top-k similar cells by embedding)
2. **Dependency expansion** (follow upstream cells to find assumptions)
3. **Recency weighting** (favor recent changes)
4. **Cell-type weighting** (markdown/comments > code execution)
5. **Hard context limit** (e.g., max 4,000 tokens of context)

Final context window is **bounded and auditable**—user sees what the agent saw.

---

## 9. Reasoning & Response

The LLM:
- **Receives only retrieved context** (no external knowledge)
- **Is instructed to**:
  - Cite specific cells (e.g., "Cell 12")
  - Say _"not enough context"_ when applicable
  - Acknowledge assumptions
  - Flag potential risks
- **Generates response** with sources

### Example Response Format

_"Q4 was removed because it contained incomplete revenue data (Cell 12). This filtering step affects the revenue trend chart in Cell 15 and the summary table in Cell 18. Assumption: Q4 is always incomplete."_

---

## 10. Failure Modes & Mitigations

| Failure | Mitigation |
|---------|-----------|
| Hallucination | Context-only prompting, explicit guardrails |
| Over-retrieval | Hard context limits (e.g., 5–10 cells max) |
| Missed dependencies | Dependency expansion algorithm |
| Ambiguous intent | User clarification prompt ("Did you mean...?") |
| Stale context | Timestamp weighting, refresh on cell change |

---

## 11. Evaluation Metrics

- **Context relevance** (manual review: did the agent retrieve the right cells?)
- **Citation accuracy** (do all claims cite a specific cell?)
- **Hallucination rate** (claims unsupported by context)
- **Coverage** (% of user queries answerable without external knowledge)
- **User trust feedback** (qualitative: do users feel it "gets" the notebook?)

---

## 12. MVP Scope (7–10 Days)

### Must Have
- ✅ Cell ingestion from notebook JSON
- ✅ Intent inference (lightweight LLM call)
- ✅ Dependency graph construction
- ✅ Vector indexing (FAISS)
- ✅ Multi-stage retrieval
- ✅ Question answering with citations
- ✅ Gradio UI for demo

### Nice to Have
- Change impact analysis ("If I modify this cell, what breaks?")
- Context thread visualization
- Export reasoning as audit trail
- Multi-notebook support

---

## 13. Tech Stack

- **Backend**: Python 3.10+
- **Vector DB**: FAISS (simple, fast, no external service)
- **LLM**: OpenAI GPT-4 (or OSS: Mistral 7B)
- **Frontend**: Gradio (rapid prototyping) or Next.js (polish)
- **Notebook format**: JSON (JupyterLab standard)
- **Dependency tracking**: Custom graph + topological sort

---

## 14. Why This Matters (Strategic Alignment)

This prototype demonstrates:

1. **Agentic reasoning** — Structured context retrieval, multi-step logic
2. **Context management** — The hard problem Hex solves
3. **Production-minded design** — Failure modes, evaluation metrics, scope discipline
4. **Deep alignment with Hex's vision** — Notebook agents that reason within bounded context

Hex's CEO has emphasized the importance of agents that respect the notebook's *structure* and *history*. This prototype shows exactly that thinking.

---

## 15. Next Steps (Post-MVP)

1. **Permissions-aware context** — Different users see different context
2. **Live notebook APIs** — Integrate with JupyterLab/VS Code
3. **Shareable context threads** — Share reasoning snapshots with teammates
4. **Multi-user reasoning** — Collaborative debugging of notebooks
5. **Feedback loop** — User can mark responses as helpful/unhelpful
6. **Integration with Hex's SDK** — If path exists

---

## 🔥 Why This Document is Powerful

This shows:
- ✅ You think in **systems** (architecture, components, flow)
- ✅ You understand **trade-offs** (what to build vs. skip)
- ✅ You respect **scope** (realistic 7-10 day MVP)
- ✅ You care about **trust and UX** (citations, failure modes, auditability)
- ✅ You're aligned with **Hex's core mission** (context-aware agents)

**This is exactly what Hex looks for in technical leaders and engineers.**
