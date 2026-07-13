# Code Review Agent

An AI code review agent built with **Gemini** and **LangGraph**, using the **ReAct pattern** (Reason → Act → Observe) to ground its feedback in real code execution and real linter output — not just a static read of the text.

## 🔗 Live Demo

**Try it here: [devansh-code-review-agent.streamlit.app](https://devansh-code-review-agent.streamlit.app/)**

Paste any Python, C++, or JavaScript code and get a structured review that's actually verified by execution and linting — not just inferred from reading the text.

> Note: uses the free tier of the Gemini API, so if you hit a rate limit during heavy testing, wait a minute and try again.

This repo also shows the project's evolution across four versions, from a plain LLM prompt to a full multi-tool agent.

## Why agentic instead of a single prompt?

A plain LLM call reviewing code can only *guess* whether something is a bug — it's reading text, not running it. This agent instead:

1. **Reasons** about what it needs to check
2. **Acts** by calling real tools (execute the code, run a linter)
3. **Observes** the actual results
4. Loops back and decides if it has enough to answer, or needs to check something else
5. Writes a final review that's grounded in verified behavior, not inference

This is the same **ReAct loop** used by production coding agents — built here from scratch with LangGraph's state/node/edge primitives, rather than a pre-built agent framework.

## Architecture

```
        ┌─────────┐
   ┌───▶│  think  │◀── Gemini reasons, decides: answer, or call a tool?
   │    └────┬────┘
   │         │
   │   has a tool call?
   │    ┌────┴─────┐
   │   yes         no
   │    │           │
   │    ▼           ▼
   │ execute_tool   END → final structured review
   │    │
   └────┘
  (loops back with the real tool result)
```

- **State** — a shared message history (`AgentState`) that accumulates every turn: the original code, the model's reasoning, tool calls, and real tool results
- **Nodes** — `think` (calls Gemini) and `execute_tool` (runs the actual Python function behind whichever tool Gemini requested)
- **Edges** — a conditional edge (`should_continue`) that checks whether Gemini's last response included a tool call, and an unconditional edge that always routes back from `execute_tool` to `think`

## Version progression

| File | What it adds |
|---|---|
| `v1_plain_review.py` | Baseline: paste code → one Gemini call → static review. No verification, no tools. |
| `v2_run_code_tool.py` | First agentic version. Adds a `run_code` tool (via `subprocess`) so the agent can actually execute the pasted code and catch real errors, timeouts, and compile failures. |
| `v3_execution_grounded_review.py` | Same tool, but adds a system instruction that forces the agent to run the code *before* answering, and structures the final output as a full review (bugs / style / performance / suggestions) grounded in the real execution result. |
| `v4_add_linter.py` | Adds a second tool, `lint_code` (pylint for Python, with cpplint/eslint hooks for C++/JS), so the agent can verify both **behavior** (does it run?) and **style** (is it well-written?) before writing one unified review. |
| `streamlit_app.py` | A web UI wrapping the v4 agent — paste code, click a button, get a live rendered review. This is what's deployed at the live demo link above. |

Each version is a complete, runnable script — you can see exactly what changed at each step by diffing them.

## Tools

- **`run_code(code, language)`** — writes the code to a temp file and executes it via `subprocess`, capturing real stdout/stderr. Supports Python (direct), JavaScript (via `node`), and C++ (compiled with `g++` first). Handles timeouts (infinite loops) and missing-interpreter errors gracefully.
- **`lint_code(code, language)`** — runs a real linter against the code and returns actual warnings/errors (unused imports, naming conventions, missing docstrings, etc.), currently most complete for Python via `pylint`.

## Setup (run locally)

**1. Clone the repo**
```bash
git clone https://github.com/DevanshAgarwal24/code-review-agent.git
cd code-review-agent
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add your Gemini API key**

Create a `.env` file in the project root:
```
GEMINI_API_KEY=your_key_here
```
Get a free key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey).

**4. Run the CLI version**
```bash
python v4_add_linter.py
```
Paste your code, type `END` on a new line, and get a structured review grounded in real execution and linting.

**5. Or run the web UI locally**
```bash
streamlit run streamlit_app.py
```

## Deployment

The live demo is deployed on **Streamlit Community Cloud**, connected directly to this repo's `main` branch — every push automatically redeploys the app.

- `requirements.txt` — Python dependencies
- `packages.txt` — system-level packages (`g++`, `nodejs`) needed for C++/JS execution and linting on the cloud container
- `GEMINI_API_KEY` is stored as an encrypted secret on Streamlit Cloud, never committed to this repo

## Requirements for full multi-language support (local)

- Python: no extra setup (built-in)
- C++: requires `g++` on your PATH
- JavaScript: requires `node` on your PATH

## Tech stack

- [Gemini API](https://ai.google.dev/) (`google-genai` SDK) — the underlying LLM
- [LangGraph](https://www.langchain.com/langgraph) — state/node/edge graph orchestration for the ReAct loop
- [Streamlit](https://streamlit.io/) — web UI and deployment
- `subprocess` — real, isolated code execution
- `pylint` — real Python linting

## Author

Devansh Agarwal — [github.com/DevanshAgarwal24](https://github.com/DevanshAgarwal24)
