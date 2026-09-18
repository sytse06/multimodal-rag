---
type: "query"
date: "2026-09-18T12:50:26.682941+00:00"
question: "Should docs/pipeline.md be fleshed out now or after a sprint?"
contributor: "graphify"
outcome: "useful"
source_nodes: ["Query Pipeline", "Provider-agnostic Model Configuration", "Gradio Queue Strategy", "Model Routing by Name Format"]
---

# Q: Should docs/pipeline.md be fleshed out now or after a sprint?

## Answer

Expanded from original query via vocab: [pipeline, epic, provider, configuration, factory, gradio]. Recommendation: outline and mark docs/pipeline.md as target-state documentation now, before Epic 8 starts, but postpone the detailed implementation rewrite until INFER-001 typed configuration and INFER-002 dependency refresh settle the APIs. Update the document incrementally: first the architecture boundary and migration notes, then provider-neutral streaming after INFER-003/004, then exact Gradio callback and queue details after INFER-005. Keep Epic 9 storage backup instructions in a separate runbook, with only an embedding-compatibility note in pipeline.md.

## Outcome

- Signal: useful

## Source Nodes

- Query Pipeline
- Provider-agnostic Model Configuration
- Gradio Queue Strategy
- Model Routing by Name Format