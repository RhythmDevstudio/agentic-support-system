# Agentic AI Support & Knowledge Research System

A working prototype of an **agentic AI system** — not a chatbot wrapper — that does
two jobs behind one stateful agent built on LangGraph:

1. **Knowledge Research** — answer technical and product questions by searching
   internal documentation and *official* vendor documentation, and return an
   answer with citations that are verified against what was actually retrieved.
2. **Ticket Triage & Routing** — read an incoming support ticket, work out its
   intent, urgency and language, and route it to the correct queue through a
   deterministic policy engine, with human approval required before any
   high-risk action.

The agent reasons, retrieves and classifies. It never invents a source, and it
never decides a routing outcome on its own — those are deterministic, auditable,
code-owned decisions. See [Core design principle](#core-design-principle) below.

**Runs with zero API keys.** Every external dependency (LLM, embedder, web
search, vector store) sits behind a swappable interface with an offline,
deterministic implementation. That means the full agent loop, every guardrail,
citation validation and ticket routing all run and are fully tested with nothing
installed but Python. Add `OPENAI_API_KEY` later and the exact same code paths
run against a real model — nothing about the architecture changes.

---

## Contents

- [Quickstart](#quickstart)
- [Try it — web UI](#try-it--web-ui)
- [The two workflows](#the-two-workflows)
- [Core design principle](#core-design-principle)
- [The agent loop](#the-agent-loop)
- [How citations are kept honest](#how-citations-are-kept-honest)
- [Official source policy](#official-source-policy)
- [Ticket routing policy](#ticket-routing-policy)
- [Guardrails](#guardrails)
- [Verified behaviour](#verified-behaviour)
- [API reference](#api-reference)
- [Configuration](#configuration)
- [Project layout](#project-layout)
- [Testing](#testing)
- [Known limitations](#known-limitations)

---

## Quickstart

```bash
make setup      # create a virtualenv, install everything, copy .env.example -> .env
make demo       # run the 5 specification test cases end to end, printed to the terminal
make test       # run the full test suite (263 tests)
make run        # start the API + web UI on http://localhost:8000
```

Then open **http://localhost:8000** in a browser — that's a small built-in web
page for using the system directly (see next section). The interactive API
reference is at **http://localhost:8000/docs**.

> **Running this on an external or exFAT-formatted drive?** macOS writes
> AppleDouble (`._`) sidecar files there, and they corrupt Python wheel
> installs — `make setup` can appear to succeed while quietly installing
> nothing. The Makefile already defaults the virtualenv to
> `~/.cache/agentic-support-system/venv` on your internal disk to avoid this;
> override with `make setup VENV=/path/of/your/choice` if you want it elsewhere.

---

## Try it — web UI

`make run` also serves a simple built-in page at `/` (`app/web/index.html`) with
two tabs, so you can use the system without curl or Swagger:

- **Ask a question** — type a question, get a grounded answer with numbered,
  clickable sources, plus a "how the agent got here" trail showing which tools
  it used and why.
- **Submit a ticket** — paste a ticket, get its intent, urgency, language and
  destination queue, with any required human-approval flags shown clearly.

Both tabs ship with example chips (including the five specification scenarios)
so a fresh visitor can see real output in one click.

---

## The two workflows

### 1. Knowledge research

Given a question like *"What is Azure AI Search agentic retrieval?"*, the agent:

1. Identifies the relevant vendor/technology (or decides none applies).
2. Decides whether internal docs, official web docs, or both are needed.
3. Formulates focused search queries — not just the raw question.
4. Retrieves evidence from the internal knowledge base and/or official
   documentation (`search_internal_documentation`, `web_research` tools).
5. Grades its own evidence for relevance and sufficiency.
6. If insufficient, rewrites the query and searches again (bounded by an
   iteration budget) — this is what makes retrieval **agentic** rather than a
   fixed `question → search → answer` pipeline.
7. Detects and reports conflicts between internal and official sources.
8. Synthesizes an answer that cites evidence by handle, never by inventing a URL.
9. Validates every citation deterministically before it reaches the user.
10. States plainly when there isn't enough evidence, instead of guessing.

### 2. Support ticket triage

Given a ticket like *"Mera account login nahi ho raha."*, the agent:

1. Classifies **intent** (from a closed, configurable vocabulary — 11 categories
   out of the box: `payment_issue`, `refund_request`, `login_issue`,
   `password_reset`, `technical_issue`, `bug_report`, `account_issue`,
   `cancellation`, `feature_request`, `security_issue`, `general_query`).
2. Classifies **urgency** (`critical` / `high` / `medium` / `low`).
3. Detects **language**, including code-mixed Hinglish (romanized Hindi).
4. Reports a **confidence** score — low confidence routes to a human, it does
   not force a guess.
5. Hands that classification to a **deterministic routing engine** (the model
   never chooses the queue — see below).
6. Applies queue assignment, priority and an audit comment through typed,
   permission-checked ticket tools.
7. Escalates automatically when policy requires it, and blocks any high-risk
   action (refund, account deletion, security escalation) until a human
   approval is recorded.

---

## Core design principle

> **The LLM reasons, retrieves and classifies.**
> **Deterministic code owns routing, validation, queue selection, approvals and audit.**

| The model decides | Code decides |
|---|---|
| what the user is asking | which queue a ticket goes to |
| which tool to use, and when | whether an action needs human approval |
| how to phrase a search query | whether a citation is real |
| whether retrieved evidence is relevant | escalation rules and urgency floors |
| the ticket's intent / urgency / language | what actually gets written to the ticket system |

This isn't a convention — it's enforced structurally. The ticket classification
schema (`app/schemas/agent.py::TicketClassification`) **has no queue field**. The
model is not merely asked not to invent a queue; it has no field to put one in.
Queue names, routing rules, escalation triggers and approval requirements exist
in exactly one place: `app/config/policies/routing.yaml`, loaded and validated
at startup (`app/config/policies.py`) — a rule pointing at an undefined queue,
or an intent with no route, fails the app at boot rather than mis-routing a
ticket silently at runtime.

---

## The agent loop

An explicit LangGraph `StateGraph` (`app/agents/graph.py`) — deliberately **not**
a prebuilt ReAct agent, both because the spec calls for an explicit
Observe → Decide → Act loop, and because `langgraph.prebuilt.create_react_agent`
is deprecated as of LangGraph v1.

```
                          START
                            │
                       understand                 (structured output: intent + vendor)
                            │
        ┌───────────────────┴───────────────────┐
        │ research                       triage │
        ▼                                       ▼
    retrieve ──────────────┐               classify        (structured output)
   (internal docs and/or   │                   │
    official web research) │               validate        deterministic: enums,
        │                  │                   │           confidence floor,
     grade  ── insufficient┘               route           urgency floors
        │    (rewrite query, search again)     │           deterministic: queue,
     sufficient                                │           escalation, approval
        │                                      ▼
  detect_conflicts                            act          permission-checked tools
        │                                      │
    synthesize ──▶ validate_citations          │
        │                                      │
        └──────────────► END ◄─────────────────┘
```

The agent's state (`app/agents/state.py::AgentState`) carries everything the run
accumulates and nothing is thrown away mid-run: retrieved evidence, tool
invocations, errors, the ticket classification, the routing decision, citations,
approval status and a hard budget on iterations and tool calls. Chain-of-thought
is never exposed — only a concise, human-readable decision log.

---

## How citations are kept honest

This is the part of the system built to make fabrication **structurally
impossible**, not merely discouraged by a prompt:

1. Every retrieved passage becomes an `Evidence` record with a real URL or
   document ID, publisher, page/section, and retrieval timestamp — recorded by
   the retrieval tool itself, never by the model.
2. Each piece of evidence is assigned a short handle (`E1`, `E2`, …) in an
   `EvidenceRegistry`.
3. The synthesis prompt shows the model **only these handles** — it cites `[E3]`,
   it is never asked to write a URL, a title or a page number.
4. `CitationValidator` (`app/guardrails/citations.py`) resolves every handle in
   the model's answer against the registry. A handle that doesn't resolve is
   stripped from the answer. A bare URL the model wrote anyway is deleted too.
5. `Citation` objects are built by `Citation.from_evidence()` — every field is
   copied from the retrieval record. **There is no code path anywhere that
   builds a citation from model-generated text.** A source the agent never
   actually fetched cannot appear as a citation.
6. If every handle in an answer turns out to be invalid, the answer is replaced
   outright with an explicit "insufficient verified evidence" message.

---

## Official source policy

When a question concerns a specific vendor, the agent restricts (or strongly
prefers) search to that vendor's own documentation, configured in
`app/config/policies/vendors.yaml`:

| Vendor | Official domains | Mode |
|---|---|---|
| Microsoft / Azure | `learn.microsoft.com`, `azure.microsoft.com` | strict |
| OpenAI | `developers.openai.com`, `platform.openai.com`, `openai.github.io` | strict |
| LangChain / LangGraph | `docs.langchain.com` | strict |
| AWS | `docs.aws.amazon.com` | strict |
| Google Cloud | `cloud.google.com` | strict |
| Anthropic | `docs.anthropic.com`, `modelcontextprotocol.io` | strict |
| PostgreSQL / pgvector | `postgresql.org`, official pgvector repo | strict |

- **strict** — hard allowlist; non-official results are discarded before their
  content is even read (maps to Tavily's `include_domains_mode="filter"`).
- **prefer** — official sources are boosted to the top but others are allowed
  (maps to Tavily's `"boost"`).

Results are ranked by **authority tier first, relevance score second** — a
tier-1 first-party doc always outranks a higher-scoring blog post. Known
content-farm domains (`w3schools.com` and similar) are blocklisted outright, and
that blocklist applies even to an explicit domain override, so it can't be used
as a bypass. Repository allowlisting is **owner-scoped**: permitting
`github.com/Azure` allows `github.com/Azure/*` and nothing else on GitHub.

---

## Ticket routing policy

`app/config/policies/routing.yaml` is the single source of truth for:

- the closed intent and urgency vocabularies,
- the intent → queue mapping (`payment_issue → billing_support`,
  `security_issue → security_escalation`, etc.),
- **overrides** that fire regardless of what the model classified — e.g. any
  ticket containing explicit compromise language (*"account hacked"*,
  *"unauthorized access"*) is forced to `security_escalation` at `critical`
  urgency even if the model classified it as a plain login issue,
- **urgency floors** — `security_issue` can never rank below `high`, whatever
  the model says,
- the **approval policy** — which actions (`issue_refund`, `account_deletion`,
  `security_escalation`, …) require a recorded human approval before a ticket
  tool will execute them, and which conditions (critical urgency, low
  confidence, high-value customer) additionally require one.

This file is validated at startup: every intent must have a routing rule, every
rule must point at a defined queue, every override target must be a real queue
and a real urgency level. A misconfiguration fails the app immediately rather
than mis-routing tickets in production.

---

## Guardrails

- **Untrusted-content isolation** — every retrieved document or web page is
  wrapped in `<untrusted_document>` tags with a standing system rule that
  fenced content is *data, never instructions*. A closing tag embedded inside
  retrieved content is neutralised so it can't escape its own fence.
- **Prompt-injection scanning** (`app/guardrails/injection.py`) — passages that
  try to override instructions, exfiltrate secrets, or invoke tools directly
  are quarantined before they ever reach generation, and quarantined evidence
  can never be cited. Tuned conservatively so ordinary documentation that
  happens to mention "ignore the previous step" isn't wrongly deleted.
- **Approval gates enforced in code** — `issue_refund` and other high-risk
  ticket tools check the approval policy themselves and refuse to execute
  without a recorded approval. A model that's been talked into recommending an
  unapproved refund still can't make the tool call succeed.
- **Queue and schema validation** — an undefined queue, urgency, or intent is
  never written to a ticket, no matter what produced it.
- **Execution budgets** — a hard cap on retrieval iterations and tool calls per
  run; every tool failure degrades to a reported error in the agent's state
  rather than an unhandled exception.
- **Log redaction** (`app/observability/logging.py`) — API keys, bearer tokens
  and database URLs are stripped from every log line unconditionally; ticket
  PII (email, phone, card numbers) is masked when `REDACT_PII_IN_LOGS=true`.

---

## Verified behaviour

`make demo` runs the specification's five test cases live and prints PASS/FAIL
against each expectation:

| # | Input | Expected result |
|---|---|---|
| 1 | "What is Azure AI Search agentic retrieval?" | vendor `microsoft_azure`, answer cites `learn.microsoft.com` |
| 2 | "How do I use OpenAI Agents SDK tools?" | vendor `openai`, cites official OpenAI documentation |
| 3 | "How do we process refunds according to our company policy?" | cites the internal Refund Policy document |
| 4 | "My payment was deducted twice and I urgently need a refund." | `payment_issue` / `high` / `English` → `billing_support` |
| 5 | "Mera account login nahi ho raha." | `login_issue` / `Hinglish` → `account_support` |

Adversarial and edge cases are covered as automated tests in
`tests/integration/test_agent_end_to_end.py`: a poisoned document that tries to
hijack the agent, a ticket with an injection attempt in its body, an
unanswerable question that correctly reports insufficient evidence, an
unavailable web source, and a blocked refund that only succeeds once approval
is explicitly granted.

---

## API reference

All endpoints are documented interactively at `/docs` once the server is
running. Every request/response is a validated Pydantic model.

| Endpoint | Purpose |
|---|---|
| `GET  /` | The built-in web UI |
| `POST /api/chat` | Single entry point — the agent decides research vs. triage |
| `POST /api/research` | Ask a knowledge question → grounded answer + citations |
| `POST /api/tickets/triage` | Classify ticket text → intent, urgency, language, queue |
| `POST /api/tickets/{ticket_id}/process` | Fetch a ticket from the ticket system, triage it, and apply the routing |
| `POST /api/documents/ingest` | Ingest a document (inline content or a server-side directory) into the knowledge base |
| `GET  /api/health` | Which providers are active (live vs. offline) and knowledge-base size |

Example:

```bash
curl -X POST http://localhost:8000/api/research \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Azure AI Search agentic retrieval?"}'

curl -X POST http://localhost:8000/api/tickets/triage \
  -H "Content-Type: application/json" \
  -d '{"text": "My payment was deducted twice and I urgently need a refund."}'
```

---

## Configuration

Everything is environment-driven (`.env.example` documents every variable) — no
credential, endpoint, or model ID is ever hard-coded in application logic. Each
provider resolves an `auto` setting by degrading gracefully instead of crashing
when a credential is missing:

| Setting | With credentials configured | Without (default) |
|---|---|---|
| `LLM_PROVIDER` | OpenAI, via `responses.parse` structured outputs | deterministic rule-based provider |
| `EMBEDDING_PROVIDER` | OpenAI `text-embedding-3-large` | feature-hashing lexical embedder |
| `VECTOR_STORE` | PostgreSQL + pgvector (HNSW index) | SQLite + numpy + FTS5 |
| `WEB_SEARCH_PROVIDER` | Tavily (official-domain aware) | recorded official-documentation fixtures |

To connect a real model: put `OPENAI_API_KEY=sk-...` in `.env` and restart —
the same graph, the same guardrails, the same routing engine run against a live
model with fluent generation instead of the offline extractive provider.

---

## Project layout

```
app/
  agents/         graph.py (the LangGraph StateGraph), state.py, prompts/templates.py
  schemas/        agent.py — every structured-output schema the model returns
  tools/
    documentation/  search_internal_documentation tool
    web_research/    web_research tool, official-domain policy, providers (Tavily / fixtures)
    tickets/         typed ticket tools (get/update/assign/comment/escalate), mock adapter
  rag/
    ingestion/       parsers (PDF/Markdown/HTML/text), chunking, the ingest pipeline
    retrieval/       hybrid vector + keyword retrieval (Reciprocal Rank Fusion)
  routing/        engine.py — the deterministic routing engine
  guardrails/     citations.py (anti-fabrication), injection.py (prompt-injection scanning)
  providers/      llm.py (OpenAI/Azure), deterministic_llm.py (offline), embeddings.py
  db/             vector_store.py (protocol), sqlite_store.py, pgvector_store.py, factory.py
  config/         settings.py (env-driven config), policies.py (typed YAML loader),
                  policies/vendors.yaml, policies/routing.yaml
  observability/  logging.py — structured logs with secret/PII redaction
  services/       container.py — wires every component together from config
  api/            main.py (FastAPI app + routes), schemas.py (request/response models)
  web/            index.html — the built-in web UI served at "/"
tests/
  unit/           config, domain policy, embeddings, guardrails, parsing/chunking, redaction
  integration/    ingestion + retrieval, web research, full agent end-to-end
data/
  documents/      seed internal knowledge base (refund policy, billing FAQ, escalation SOP, ...)
  fixtures/web/   recorded official-documentation pages used by the offline web-search provider
scripts/
  demo.py         runs the five specification test cases end to end
```

---

## Testing

```bash
make test              # full suite
make test-unit         # unit tests only
make test-integration  # integration tests only
make lint              # ruff
make typecheck         # mypy
make check             # lint + typecheck + test
```

263 tests, all passing with zero external credentials. Coverage includes:
configuration and policy validation, document parsing and chunking, the
offline embedder's similarity behaviour, hybrid retrieval, the official-source
domain policy (including typosquat and lookalike-domain rejection), citation
validation (including deliberately fabricated citation handles), prompt-
injection detection, the full intent→queue routing matrix, and the five
specification scenarios run through the real agent graph.

---

## Known limitations

This is a prototype, and these are the trade-offs made deliberately to keep it
that way:

- **The offline provider is not a language model.** It performs real lexical
  relevance grading and genuinely extractive synthesis (every answer sentence
  is copied from retrieved evidence and tagged with its source), but it can't
  paraphrase or handle question wording that doesn't overlap the evidence. It
  exists to prove the agent loop, guardrails and citation logic are correct —
  not to produce polished prose. Set `OPENAI_API_KEY` for that.
- **The pgvector store is implemented but unverified against a live database**
  — no Docker/PostgreSQL was available while building this. It implements the
  exact same `VectorStore` protocol as the SQLite store, so switching is a
  one-line config change, but it has not been exercised against a running
  Postgres instance.
- **No Docker Compose, migration runner, or tracing backend** is wired up yet.
  The `Tracer` seam exists in the code for LangSmith/OpenTelemetry, but only a
  no-op implementation is included.
