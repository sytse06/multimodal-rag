# Query Pipeline

## Overview

The query pipeline converts a natural language support question into a cited answer and, optionally, a KB article. It has two phases:

1. **Retrieval** — embed the question, search Weaviate, return ranked `SearchResult` objects.
2. **Generation** — pass the ranked chunks to an LLM to produce a `CitedAnswer` with inline links.

The KB article editorial workflow is a downstream extension of phase 2: the same `CitedAnswer` and `list[SearchResult]` from state are fed to a second LLM call that generates a structured article.

## Target Architecture for Epic 8

The pipeline is being migrated from UI-owned, blocking model calls to a provider-neutral
inference boundary. Gradio should select a validated chat provider and model, while the
application layer constructs the LangChain `BaseChatModel` through one centralized
factory. Embeddings remain independently configured so changing the chat provider never
changes the vectors already stored in Weaviate.

The target generation path exposes streaming progress and completion events. Gradio
adapts those events into cumulative chatbot and article-draft updates; it does not
construct provider clients or contain provider-specific routing. Citation links are
constructed only after the final response has been accumulated, because partial chunks
cannot be reliably rewritten.

This document currently describes the implementation being migrated. References to
`_make_llm`, model-name routing, blocking `invoke()`, and final-only state updates are
legacy behavior and must not be extended. The exact typed settings, provider registry,
event models, and callback signatures will be finalized in INFER-001 through INFER-005.

---

## Data Models (`models/query.py`)

### `SearchResult`

Represents one retrieved chunk from Weaviate. Key computed properties:

| Property | Logic |
|---|---|
| `citation_url` | For video chunks: appends `?t={seconds}s` or `&t={seconds}s` depending on whether the base URL already contains `?`. For web chunks: returns `source_url` unchanged. |
| `citation_label` | For video: `"{source_name} @ MM:SS"`. For web with section heading: `"{source_name} — {section_heading}"`. Otherwise: `source_name`. |
| `citation_markdown` | `[label](url) (score%)` — used by callers that need inline markdown. |

The `?` vs `&` separator in `citation_url` handles both short-form `youtu.be/{id}` URLs (no existing query string) and standard `youtube.com/watch?v={id}` URLs (query string already present). Without this, a second `?` would produce a malformed URL.

### `Citation`

A serialisable snapshot of one source, stored on `CitedAnswer`. Contains `label`, `url`, `relevance_score`, and `source_type`. These are copied from `SearchResult` properties at generation time so that downstream consumers (e.g. the KB article workflow) do not need to re-derive URLs from raw chunk data.

### `CitedAnswer`

```python
class CitedAnswer(BaseModel):
    answer: str          # markdown, with [1] refs replaced by clickable links
    citations: list[Citation]
```

This is the unit passed through Gradio state between the chat pipeline and the review workflow.

---

## Query Pipeline: Step by Step

### 1. User submits a question

The `user_submit` handler in `app.py` is triggered by the textbox submit or the Submit
button. It resolves the explicit provider/model selection, retrieves sources, and
consumes `stream_cited_answer()`:

```python
for event in stream_cited_answer(
    question, results, llm, provider=selection.provider, model=selection.model
):
    yield cumulative_chat_history(event), event
```

Retrieval and generation progress are yielded separately. The chatbot displays raw
cumulative text during generation; citation links are formatted only when the complete
event arrives. A Stop action cancels the active Gradio event.

After completion, the following items are written to Gradio state:
- `last_answer_state` — the `CitedAnswer`
- `last_results_state` — the `list[SearchResult]`
- `last_question_state` — the raw question string
- `last_selection_state` — the selected `ModelSelection`

Provider, model, completion status, usage, latency, and safe error context come from
typed inference events rather than provider-specific Gradio logic.

### 2. Retrieval (`retriever.py`)

`retrieve()` fetches a candidate pool of `top_k * 4` results from Weaviate, then applies a rebalancing filter to ensure video content is not crowded out by web content:

- At least `ceil(top_k / 2)` slots are reserved for video chunks.
- The remaining slots go to web chunks.
- The final `top_k` list is re-sorted by `relevance_score` descending.

Weaviate returns cosine distance (0 = identical, 2 = opposite). `_distance_to_score` converts this to a 0–1 relevance score via `max(0.0, 1.0 - distance)`.

When `SupportChunk` is restored from the fixed Weaviate snapshot described in Epic 9,
the query embedding provider, model, and vector dimension must remain compatible with
the snapshot. Snapshot creation and restoration are operational concerns documented
outside this pipeline; this compatibility rule is the pipeline-level dependency.

### 3. Context formatting (`retriever.py`)

`format_context(results)` serialises the ranked chunks into a numbered context block:

```
[1] {citation_label}
{chunk text}

[2] {citation_label}
{chunk text}
```

This numbered format is what enables the LLM to write `[1]`, `[2]` etc. as citations.

### 4. LLM call and system prompt (`generator.py`)

`generate_cited_answer` sends two messages to the LLM:

- `SystemMessage` — `SYSTEM_PROMPT`
- `HumanMessage` — `USER_TEMPLATE` with `{context}` and `{question}` interpolated

**`SYSTEM_PROMPT` rules:**

1. Answer using ONLY the provided source chunks — do not invent information.
2. Cite sources using bracket notation `[1]`, `[2]` matching the chunk numbers.
3. If sources are insufficient, say so.
4. Keep answers concise and actionable for support staff.
5. Use markdown formatting.

### 5. Citation building and inline link substitution

After the LLM returns its raw answer text, two post-processing steps run:

**`_build_citations(results)`** — creates a `list[Citation]` directly from the `SearchResult` objects. URLs are taken from `r.citation_url` (the computed property, with timestamp appended). This happens independently of the LLM output — the LLM cannot alter or fabricate citation URLs.

**`_replace_refs_with_links(raw_answer, results)`** — iterates through results 1-indexed and does a string replacement: `[1]` → `[citation_label](citation_url) (score%)`. This transforms LLM-generated bracket refs into rendered markdown links.

Because URL construction happens in `_build_citations` and `_replace_refs_with_links` rather than being delegated to the LLM, citation URLs are guaranteed to be accurate — the LLM can only influence which numbers appear in the answer text, not where those numbers point.

### 6. Chat display

`_format_citations_block(answer)` appends a `---` divider and a `**Sources:**` list below the answer markdown before writing to the chatbot. Video sources get a film-strip icon, web sources get a page icon.

---

## KB Article Editorial Workflow

The walkthrough is triggered by the "Review & save as article" button. It uses a `gr.Walkthrough` with 4 steps rendered inside `walkthrough_col`, a column that starts hidden. The `input_row` (question textbox + buttons) hides when the walkthrough opens and restores on cancel.

### Layout transition

`enter_review` runs `queue=False` and makes the following changes atomically:

- `input_row`: `visible=False`
- `walkthrough_col`: `visible=True`
- `chatbot`: height reduced from 500 to 300 (chat stays visible above the walkthrough)
- `walkthrough`: reset to step 1
- `step1_answer`, `step1_citations`: populated from `_format_step1(answer)`

If `last_answer_state` is `None` (no question asked yet), all outputs return `gr.update()` with no changes — the button is a no-op.

---

### Step 1 — Review answer

**Handler:** `enter_review` (called by `review_btn.click`)
**Queue:** `False`

`_format_step1(answer)` splits the `CitedAnswer` into two markdown strings:
- `answer.answer` — the full answer with inline citation links
- A bullet list of `[label](url) (score%)` entries, one per citation

These populate `step1_answer` and `step1_citations` as separate `gr.Markdown` components.

The Cancel button calls `cancel_review` (`queue=False`), which reverses the layout transition: shows `input_row`, hides `walkthrough_col`, restores chatbot height to 500.

---

### Step 2 — Inspect sources

**Handler:** `go_to_step2(results)` (called by `next1_btn.click`)
**Queue:** `False`

`_format_step2(results)` renders the raw chunk text for every `SearchResult`. Each block shows the 1-indexed number, icon, label, relevance score, and the full `r.text` — the unprocessed content retrieved from Weaviate. Blocks are separated by `---` dividers.

Sources are loaded lazily here on first visit from `last_results_state` — they are not pre-populated when the walkthrough opens.

---

### Step 3 — Edit draft

**Handler:** `go_to_step3(answer, results, model, question)` (called by `next2_btn.click`)
**Queue:** `True` (default — not set to `False`)

This step consumes `stream_kb_article()` with:
- `answer` — the `CitedAnswer` from state (for citations and fallback text)
- `llm` — a fresh instance from the selected provider/model factory
- `results` — the `list[SearchResult]` from state (for full source chunk text)
- `question` — the original question string from state

The draft is progressively placed into `article_editor`, a 20-line `gr.Textbox` the
user can edit freely before proceeding. The final article remains the editable value
used by Step 4.

---

### Step 4 — Save

**Handler:** `go_to_step4(article)` (called by `next3_btn.click`)
**Queue:** `False`

`go_to_step4` scans the article text for the first line starting with `# ` and strips the prefix to pre-populate `title_input`. If the LLM generated a proper H1 heading (which `KB_ARTICLE_PROMPT` encourages via its markdown formatting rule), the title field is ready without user input.

`do_save` calls `save_kb_article(title, body)`, which:
1. Falls back to `"Untitled Article"` if the title is blank.
2. Creates `kb_output/` if it does not exist.
3. Slugifies the title via `_slugify` (lowercased, non-alphanumeric stripped, spaces/hyphens collapsed, max 60 characters).
4. Formats a timestamp as `YYYYMMDD-HHMMSS`.
5. Writes `kb_output/{slug}-{timestamp}.md` with a `# {title}` prepended to the body.
6. Returns the file path string, which is displayed in `save_msg`.

---

## KB Article Generation (`generator.py`)

### `generate_kb_article` inputs

```python
def generate_kb_article(
    answer: CitedAnswer,
    llm: BaseChatModel,
    results: list[SearchResult] | None = None,
    question: str = "",
) -> str:
```

### User message structure

The human message sent to the LLM is assembled in parts:

```
## Question

{question}

## Source Material

### Source [1]: {citation_label}

{chunk text}

### Source [2]: {citation_label}

{chunk text}
```

If `results` is `None` or empty, `## Source Material` falls back to `answer.answer` as the source body. In practice the walkthrough always passes `results` from state.

### `KB_ARTICLE_PROMPT` rules

1. Write for support staff who may not have seen the original question.
2. Open with a short summary sentence stating what the article covers.
3. Use markdown: headers, bullet points, numbered steps where appropriate.
4. Include all specific detail from the sources — steps, settings, menu paths, keyboard shortcuts, field names. Do not summarise away detail.
5. Do not invent information not present in the sources.
6. Output raw markdown directly — do NOT wrap in a code block.

Rule 6 exists because several LLMs wrap markdown output in triple-backtick fences when prompted to produce markdown. `_strip_code_fence` handles the cases where they do it anyway.

### `_strip_code_fence` post-processing

```python
def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text[text.index("\n") + 1:] if "\n" in text else ""
        if text.endswith("```"):
            text = text[:text.rfind("```")].rstrip()
    return text
```

Strips the opening fence line (which may include a language tag like ` ```markdown`) and the closing ` ``` `. Only fires if the response actually starts with a backtick fence — no-ops on clean responses.

### Sources section appended programmatically

After `_strip_code_fence`, a `## Sources` section is built from `answer.citations` and appended:

```python
if answer.citations:
    source_lines = "\n".join(f"- [{c.label}]({c.url})" for c in answer.citations)
    article = article.rstrip() + f"\n\n## Sources\n\n{source_lines}"
```

The LLM is not asked to generate this section. This is deliberate: `Citation.url` contains the correct timestamped video URLs (constructed by `SearchResult.citation_url` at retrieval time). Delegating URL generation to the LLM risks hallucinated or non-timestamped URLs.

---

## Queue Strategy

Gradio queues handlers by default. Handlers that do no I/O should bypass the queue to avoid being blocked by in-progress LLM calls.

| Handler | Queue | Reason |
|---|---|---|
| `user_submit` | `True` (default) | Retrieval and streaming generation |
| `enter_review` | `False` | Pure state read + UI update |
| `cancel_review` | `False` | Pure UI update |
| `go_to_step2` | `False` | Formats already-fetched state |
| `back2_btn` lambda | `False` | Step navigation only |
| `go_to_step3` | `True` (default) | Streaming article generation |
| `back3_btn` lambda | `False` | Step navigation only |
| `go_to_step4` | `False` | H1 extraction from string |
| `back4_btn` lambda | `False` | Step navigation only |
| `do_save` | `True` (default) | File I/O (acceptable) |

Without `queue=False` on navigation handlers, clicking "Next" or "Back" while an LLM call is in progress would queue behind it, producing a multi-second delay on what should be an instant UI update.

The Epic 8 target keeps non-I/O navigation handlers outside the queue and changes the
I/O handlers to consume streaming generators or async generators. Final citation
construction and state persistence happen when the stream completes; provider errors
must produce a safe user-facing event and retain diagnostic context for logs.

---

## Legacy Model Routing

`_make_llm(model_name, settings)` routes based on name format:

- **Bare name** (no `/`) — treated as an Ollama model, returns `ChatOllama`.
- **`provider/model` format** — treated as OpenRouter, returns `ChatOpenAI` pointed at `settings.openrouter_base_url`.

Both are configured at `temperature=0.3`. The same routing is used for both `generate_cited_answer` and `generate_kb_article` calls — they use whichever model is selected in the dropdown at call time.

The model dropdown is populated at startup by combining any Ollama model found in `settings.llm_model` with the hardcoded `OPENROUTER_MODELS` list. The configured `settings.llm_model` is used as the default selection if it appears in the list.

This routing is retained here only as a migration reference. Epic 8 replaces name-based
routing and the hardcoded model list with validated provider/model registry entries for
OpenRouter, OpenAI, Gemini, and Ollama. Gradio receives only combinations that are
configured and usable; it does not infer a provider from the model name.
