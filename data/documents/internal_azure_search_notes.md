# Internal Engineering Notes: Azure AI Search Evaluation

**Document owner:** Platform Engineering
**Version:** 0.4 (DRAFT)
**Last reviewed:** 2025-03-11
**Status:** Superseded in part — see the conflict note below

> **Note for the support knowledge base:** this is an internal engineering
> evaluation note, not vendor documentation. Where it disagrees with current
> Microsoft Learn documentation, the vendor documentation is authoritative for
> current product behaviour. This document is retained because it records our own
> deployment decisions, which vendor documentation cannot tell us.

## Why this document exists

We evaluated Azure AI Search in early 2025 as the retrieval backend for the support
knowledge platform. These notes record what we concluded at the time and which
choices we committed to.

## Our deployment decisions

These are internal decisions and remain current:

- We use one search index per product line, not a shared index with a filter.
- Index refresh runs nightly at 02:00 UTC via a scheduled indexer.
- Semantic ranking is enabled only for the customer-facing support index, because of
  its per-query cost.
- We cap retrieved passages at 8 per query before sending them to the model.

## Notes on agentic retrieval (POTENTIALLY OUTDATED)

At the time of evaluation we recorded the following understanding:

- Agentic retrieval was described as a preview capability that breaks a complex
  question into multiple subqueries and runs them in parallel against the index.
- We noted that it appeared to require a separate knowledge agent resource to be
  provisioned alongside the search service.
- We recorded that query planning used a chat-completions style model call, and that
  the response returned a unified result set rather than per-subquery results.

**Conflict note:** the details above were captured in March 2025 against a preview
release. Product behaviour, resource naming and API surface have changed since. Any
customer-facing or architectural answer about how Azure AI Search agentic retrieval
works today must be grounded in current Microsoft Learn documentation, not in this
file. Cite this document only for our own deployment decisions.

## Cost observations from the evaluation

Recorded for internal planning only, and now out of date:

- Semantic ranker was billed per 1,000 queries above a monthly free allocation.
- Storage cost was dominated by vector fields rather than by text fields.
- We estimated that shortening embeddings from 3072 to 1536 dimensions halved index
  storage with a small measured recall loss on our own evaluation set.

## Open questions carried forward

- Whether to move to a shared index with security filters as the product line count
  grows.
- Whether nightly refresh is frequent enough now that documentation changes more
  often.
