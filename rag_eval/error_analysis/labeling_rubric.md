# Manual Error Analysis — Labeling Rubric

Status: template only. No labels have been assigned; this file defines the
categories a rater applies once real model outputs exist.

Apply exactly one `human_label` per row in `error_log_template.csv` (add a
short free-text note in `rater_notes` for anything ambiguous or multi-label
in practice — do not stretch a single category to cover a mixed case).

| Label | Definition |
|---|---|
| `grounded_correct` | Answer is correct and clearly supported by the retrieved/provided context. |
| `grounded_incorrect` | Answer is wrong, but it is clearly derived from the provided context (context itself may be misleading, wrong, or insufficient). |
| `ungrounded_correct` | Answer is correct but does not draw on the provided context — consistent with parametric memory rather than grounding (relevant to Evidence Ignorance). |
| `ungrounded_hallucinated` | Answer is wrong and not traceable to either the context or a plausible parametric fact — fabricated content. |
| `partial_grounding` | Answer mixes content from the context with unsupported additions or parametric claims. |
| `refused_abstained` | Model explicitly declines to answer or states it lacks sufficient information. |
| `retrieval_failure_not_model_fault` | The retrieved context did not contain the needed evidence (e.g., Retrieval Blindness/Noise conditions) — the generation itself may be reasonable given what it was given. |
| `other` | Anything not covered above — explain in `rater_notes`. |

## Rater instructions

1. Read the question, the exact context passages that were actually sent to the model for that call (from the matching `raw_calls/.../trial_N.json`), the generated answer, and the reference answer.
2. Assign one label from the table above.
3. Record `rater_id` and `date_labeled` for every row — this is required for reproducibility and for computing inter-rater agreement if a second rater reviews a subsample.
4. Sample size and selection method (e.g., all errors below some correctness threshold, or a random stratified sample per model/condition) is still an open Phase 1 decision (#8) — do not start labeling until that is confirmed.
