# ConstraintSuite Dataset Construction Guide

**Version:** v0 (Negation)
**Last Updated:** January 2026
**Status:** In Development

---

## 1. Executive Summary (TL;DR)

### What We're Building
**ConstraintSuite-v0 (Negation)**: A diagnostic dataset for testing whether cross-encoder rerankers can handle negation constraints in queries.

### Why We're Building It
To answer a mechanistic interpretability question:
> Does a reranker cross-encoder rely on **specific internal components** (layers, heads, MLPs) to handle negation, and can we find those components using activation patching and ablation?

### How It Works
Each example is a **pairwise ranking test** where:
- A query contains a negation constraint (e.g., "without selenium")
- Two documents are compared: one satisfies the constraint, one violates it
- The model is "correct" if it scores the satisfying document higher

### The Scientific Question
> Are constraint circuits **causally localizable** and do they **drift/get overwritten** during sequential fine-tuning?

---

## 2. The Core Idea

### Pairwise Ranking Test

```
Query: "python web scraping without selenium"
                        |
                        v
        +---------------+---------------+
        |                               |
    doc_pos                          doc_neg
    (requests + BeautifulSoup        (Selenium WebDriver
     tutorial, no mention of          tutorial, explains
     selenium)                        browser automation)
        |                               |
        +---------------+---------------+
                        |
                        v
              Model scores both:
              s(q, doc_pos) and s(q, doc_neg)
                        |
                        v
    CORRECT if: s(q, doc_pos) > s(q, doc_neg)
```

### Key Insight
By keeping **topical relevance roughly equal** between doc_pos and doc_neg, the **negation constraint becomes the deciding factor**. This isolation is critical for mechanistic analysis.

### Evaluation Metrics

1. **Pairwise Accuracy**: % of examples where `score(doc_pos) > score(doc_neg)`
2. **Score Gap (Δ)**: `score(doc_pos) - score(doc_neg)` — magnitude of preference
3. **Query Sensitivity (Δ_sens)**: `Δ(q_neg) - Δ(q_base)` — how much adding negation changes preference

---

## 3. Three Dataset Slices

We construct **three distinct slices** because they answer different questions and have different utility for mechanistic analysis:

| Slice | Description | Example doc_pos | Example doc_neg | Best For |
|-------|-------------|-----------------|-----------------|----------|
| **MinPairs** | Near-identical docs with surgical single edit | "This recipe contains **no** peanuts..." | "This recipe contains peanuts..." | Cleanest mechanistic signal; patching/ablation localization |
| **ExplicitMention** | Both docs mention Y, but doc_pos negates it | "peanut-free stir fry, safe for allergies" | "Thai peanut noodles with peanut sauce" | Balanced lexical pressure; tests semantic understanding |
| **Omission** | doc_pos never mentions Y at all | "Stir fry with vegetables and tofu..." (no mention of peanuts) | "Thai peanut noodles recipe..." | Real-world stress test; exposes term-match vs constraint conflict |

### Why Three Slices?

**MinPairs** gives the cleanest signal for mechanistic work because the behavioral difference is most likely attributable to negation handling alone.

**ExplicitMention** balances lexical overlap — both docs contain the forbidden term Y, so a naive "keyword detector" circuit won't suffice.

**Omission** tests real-world scenarios but conflates negation handling with term-match heuristics (doc_neg contains query term Y, doc_pos doesn't).

### Recommendation for Mechanistic Analysis
Use **MinPairs + ExplicitMention** as your primary patching/ablation probe set. Use **Omission** as an ecological stress test and to characterize failure modes.

---

## 4. Data Schema (JSONL Format)

Each line in the JSONL file is one pairwise ranking example:

```json
{
  "id": "negation_explicit_msmarco_000123",
  "suite": "negation_explicit",
  "source": {
    "corpus": "msmarco-passage",
    "qid": "123456",
    "doc_pos_id": "msmarco:7890123",
    "doc_neg_id": "msmarco:7890456",
    "retrieval": {
      "method": "bm25",
      "index": "msmarco-v1-passage",
      "k_pool": 200,
      "rank_pos_in_pool": 14,
      "rank_neg_in_pool": 9
    }
  },
  "query": {
    "base": "python web scraping selenium",
    "neg": "python web scraping without selenium",
    "template": "WITHOUT_Y"
  },
  "constraint": {
    "type": "exclude",
    "y": "selenium",
    "negation_marker": "without",
    "y_surface_forms": ["selenium", "webdriver", "Selenium"]
  },
  "docs": {
    "pos": {
      "id": "msmarco:7890123",
      "title": "Web Scraping with Python",
      "text": "Use requests library to fetch pages and BeautifulSoup to parse HTML. This approach works well for static sites without JavaScript rendering requirements..."
    },
    "neg": {
      "id": "msmarco:7890456",
      "title": "Browser Automation with Selenium",
      "text": "Selenium WebDriver allows you to automate Chrome, Firefox, and other browsers. Install selenium with pip and download the appropriate WebDriver executable..."
    }
  },
  "labels": {
    "pairwise_preference_for_query_neg": "pos_over_neg"
  },
  "tags": {
    "difficulty": "medium",
    "doc_pos_mentions_y": false,
    "doc_neg_mentions_y": true,
    "y_negated_in_doc_pos": false,
    "lexical_overlap_bin": "high",
    "doc_length_bin": "medium",
    "negation_explicitness": "explicit"
  }
}
```

### Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique identifier: `{suite}_{corpus}_{index}` |
| `suite` | string | One of: `negation_minpairs`, `negation_explicit`, `negation_omission`, `controls_nonflip`, `controls_adversarial` |
| `source.corpus` | string | Source corpus (e.g., `msmarco-passage`) |
| `source.retrieval` | object | How candidates were retrieved (for reproducibility) |
| `query.base` | string | Original query without negation |
| `query.neg` | string | Query with negation added |
| `query.template` | string | Which template was used (`WITHOUT_Y`, `EXCLUDING_Y`, `NOT_ABOUT_Y`) |
| `constraint.y` | string | The forbidden entity/term |
| `constraint.y_surface_forms` | list | All surface forms to check for Y |
| `docs.pos` | object | Document that satisfies the constraint |
| `docs.neg` | object | Document that violates the constraint |
| `labels.pairwise_preference_for_query_neg` | string | Always `pos_over_neg` for main dataset |
| `tags.difficulty` | string | `easy`, `medium`, `hard` |
| `tags.doc_pos_mentions_y` | bool | Whether doc_pos mentions Y at all |
| `tags.lexical_overlap_bin` | string | `low`, `medium`, `high` |

---

## 5. Construction Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        CONSTRUCTION PIPELINE                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  [1] Load Corpus    [2] Generate Queries    [3] BM25 Retrieval         │
│       (MS MARCO)         (add negation)          (k=200 pool)          │
│           │                    │                      │                 │
│           v                    v                      v                 │
│       queries.tsv         q_base + q_neg         candidate_pool        │
│                                                                         │
│  [4] Mine Pairs     [5] Filter + QA         [6] Tag Difficulty         │
│       (pos/neg)         (similarity,            (overlap, length,      │
│                          dedup, validity)        mentions_y)           │
│           │                    │                      │                 │
│           v                    v                      v                 │
│       raw_pairs           filtered_pairs         tagged_pairs          │
│                                                                         │
│  [7] Sample Gold    [8] Evaluate Baselines                             │
│       (30-60)            (MiniLM, BGE)                                  │
│           │                    │                                        │
│           v                    v                                        │
│       gold.jsonl          baseline_results.json                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Stage 1: Load Corpus

We use **MS MARCO passages** as the primary corpus (well-indexed, standard benchmark).

```python
from pyserini.search.lucene import LuceneSearcher

# Load prebuilt MS MARCO passage index
searcher = LuceneSearcher.from_prebuilt_index("msmarco-v1-passage")

# Or load MS MARCO queries
import ir_datasets
dataset = ir_datasets.load("msmarco-passage/train")
queries = {q.query_id: q.text for q in dataset.queries_iter()}
```

### Stage 2: Generate Negated Queries

Templates for negation:

| Template | Pattern | Example |
|----------|---------|---------|
| `WITHOUT_Y` | `{topic} without {Y}` | "python web scraping without selenium" |
| `EXCLUDING_Y` | `{topic} excluding {Y}` | "best beaches excluding Goa" |
| `NOT_ABOUT_Y` | `{topic} not about {Y}` | "Tesla news not about Elon Musk" |

```python
TEMPLATES = {
    "WITHOUT_Y": "{topic} without {y}",
    "EXCLUDING_Y": "{topic} excluding {y}",
    "NOT_ABOUT_Y": "{topic} not about {y}",
}

def generate_negated_query(base_query: str, y: str, template: str = "WITHOUT_Y") -> str:
    """Generate a negated query from a base query and forbidden term."""
    return TEMPLATES[template].format(topic=base_query, y=y)
```

### Stage 3: BM25 Retrieval

Retrieve a candidate pool of 200 documents per query:

```python
from pyserini.search.lucene import LuceneSearcher

def retrieve_candidates(
    searcher: LuceneSearcher,
    query: str,
    k: int = 200
) -> list[dict]:
    """Retrieve top-k candidates using BM25."""
    hits = searcher.search(query, k=k)
    candidates = []
    for rank, hit in enumerate(hits):
        doc = searcher.doc(hit.docid)
        candidates.append({
            "doc_id": hit.docid,
            "text": doc.raw(),  # or parse JSON and extract text
            "bm25_rank": rank,
            "bm25_score": hit.score,
        })
    return candidates
```

### Stage 4: Mine Pairs

Core pair mining logic:

```python
import re

NEG_MARKERS = [r"\bno\b", r"\bwithout\b", r"\bfree of\b", r"\bexclude[sd]?\b", r"\bnot\b"]

def contains_y(text: str, y_forms: list[str]) -> bool:
    """Check if text contains any surface form of Y."""
    text_lower = text.lower()
    return any(form.lower() in text_lower for form in y_forms)

def y_is_negated_nearby(text: str, y: str, window: int = 40) -> bool:
    """Check if Y appears near a negation marker."""
    text_lower = text.lower()
    y_lower = y.lower()
    idx = text_lower.find(y_lower)
    if idx == -1:
        return False
    # Check window around Y for negation markers
    left = max(0, idx - window)
    right = min(len(text_lower), idx + len(y_lower) + window)
    span = text_lower[left:right]
    return any(re.search(pat, span) for pat in NEG_MARKERS)

def mine_pair(
    candidates: list[dict],
    y_forms: list[str]
) -> tuple[dict, dict] | None:
    """
    Mine a (doc_pos, doc_neg) pair from candidates.

    - doc_neg: contains Y affirmatively (not negated)
    - doc_pos: either omits Y OR contains Y in negated form

    Returns None if no valid pair found.
    """
    # Find candidates that violate constraint (contain Y affirmatively)
    violators = [
        c for c in candidates
        if contains_y(c["text"], y_forms)
        and not y_is_negated_nearby(c["text"], y_forms[0])
    ]

    # Find candidates that satisfy constraint
    satisfiers = [
        c for c in candidates
        if not contains_y(c["text"], y_forms)
        or y_is_negated_nearby(c["text"], y_forms[0])
    ]

    if not violators or not satisfiers:
        return None

    # Pick best-ranked violator as doc_neg
    doc_neg = violators[0]

    # Pick satisfier with closest BM25 rank (to match topical relevance)
    doc_pos = min(
        satisfiers,
        key=lambda c: abs(c["bm25_rank"] - doc_neg["bm25_rank"])
    )

    return doc_pos, doc_neg
```

### Stage 5: Filter and QA

Apply filters to ensure quality:

```python
def passes_filters(
    doc_pos: dict,
    doc_neg: dict,
    query_base: str,
    config: dict
) -> bool:
    """Check if a pair passes all quality filters."""

    # 1. Document length ratio
    len_pos = len(doc_pos["text"])
    len_neg = len(doc_neg["text"])
    ratio = max(len_pos, len_neg) / max(min(len_pos, len_neg), 1)
    if ratio > config["max_doc_length_ratio"]:
        return False

    # 2. Minimum document length
    if len_pos < config["min_doc_length"] or len_neg < config["min_doc_length"]:
        return False

    # 3. Lexical overlap with query (both docs should be on-topic)
    query_terms = set(query_base.lower().split())
    pos_terms = set(doc_pos["text"].lower().split())
    neg_terms = set(doc_neg["text"].lower().split())

    overlap_pos = len(query_terms & pos_terms) / len(query_terms)
    overlap_neg = len(query_terms & neg_terms) / len(query_terms)

    if overlap_pos < config["min_lexical_overlap"]:
        return False
    if overlap_neg < config["min_lexical_overlap"]:
        return False

    return True
```

### Stage 6: Tag Difficulty

Assign difficulty tags for stratification:

```python
def compute_tags(
    doc_pos: dict,
    doc_neg: dict,
    y_forms: list[str],
    config: dict
) -> dict:
    """Compute metadata tags for a pair."""

    # Lexical overlap between docs
    pos_tokens = set(doc_pos["text"].lower().split())
    neg_tokens = set(doc_neg["text"].lower().split())
    jaccard = len(pos_tokens & neg_tokens) / len(pos_tokens | neg_tokens)

    # Bin lexical overlap
    bins = config["lexical_overlap_bins"]
    if jaccard < bins[0]:
        overlap_bin = "low"
    elif jaccard < bins[1]:
        overlap_bin = "medium"
    else:
        overlap_bin = "high"

    # Document length bin
    avg_len = (len(doc_pos["text"]) + len(doc_neg["text"])) / 2
    len_bins = config["doc_length_bins"]
    if avg_len < len_bins[0]:
        length_bin = "short"
    elif avg_len < len_bins[1]:
        length_bin = "medium"
    else:
        length_bin = "long"

    # Does doc_pos mention Y?
    mentions_y = contains_y(doc_pos["text"], y_forms)

    # Difficulty heuristic
    if overlap_bin == "high" and not mentions_y:
        difficulty = "hard"  # High overlap, Y absent = hard to distinguish
    elif overlap_bin == "low":
        difficulty = "easy"  # Low overlap = topic difference helps
    else:
        difficulty = "medium"

    return {
        "lexical_overlap_bin": overlap_bin,
        "doc_length_bin": length_bin,
        "doc_pos_mentions_y": mentions_y,
        "difficulty": difficulty,
    }
```

### Stage 7: Sample Gold Set

Sample 30-60 examples for manual verification:

```python
def sample_gold_set(
    examples: list[dict],
    target_size: int = 50,
    oversample_hard: bool = True
) -> list[dict]:
    """Sample a gold set, oversampling hard examples."""
    import random

    easy = [e for e in examples if e["tags"]["difficulty"] == "easy"]
    medium = [e for e in examples if e["tags"]["difficulty"] == "medium"]
    hard = [e for e in examples if e["tags"]["difficulty"] == "hard"]

    if oversample_hard:
        # 15% easy, 35% medium, 50% hard
        n_easy = int(target_size * 0.15)
        n_medium = int(target_size * 0.35)
        n_hard = target_size - n_easy - n_medium
    else:
        n_easy = n_medium = n_hard = target_size // 3

    gold = (
        random.sample(easy, min(n_easy, len(easy))) +
        random.sample(medium, min(n_medium, len(medium))) +
        random.sample(hard, min(n_hard, len(hard)))
    )

    return gold
```

### Stage 8: Evaluate Baselines

Score pairs with cross-encoder rerankers:

```python
from sentence_transformers import CrossEncoder

def evaluate_reranker(
    model_name: str,
    examples: list[dict]
) -> dict:
    """Evaluate a reranker on the dataset."""
    reranker = CrossEncoder(model_name)

    correct = 0
    score_gaps = []

    for ex in examples:
        q_neg = ex["query"]["neg"]
        doc_pos_text = ex["docs"]["pos"]["text"]
        doc_neg_text = ex["docs"]["neg"]["text"]

        # Score both pairs
        pairs = [(q_neg, doc_pos_text), (q_neg, doc_neg_text)]
        scores = reranker.predict(pairs)

        s_pos, s_neg = scores[0], scores[1]
        gap = float(s_pos - s_neg)
        score_gaps.append(gap)

        if s_pos > s_neg:
            correct += 1

    return {
        "model": model_name,
        "pairwise_accuracy": correct / len(examples),
        "mean_score_gap": sum(score_gaps) / len(score_gaps),
        "score_gaps": score_gaps,
    }

# Usage
results_minilm = evaluate_reranker(
    "cross-encoder/ms-marco-MiniLM-L6-v2",
    examples
)
results_bge = evaluate_reranker(
    "BAAI/bge-reranker-base",
    examples
)
```

---

## 6. Size Targets

| Dataset | Target Size | Purpose |
|---------|-------------|---------|
| **Main set** | 1,000 - 5,000 | Stable evaluation averages, trend analysis |
| **Gold set** | 30 - 60 | Manual verification, qualitative analysis, patching demos |
| **Controls (non-flip)** | 500 | Sanity checks — transformations that shouldn't change ranking |
| **Controls (adversarial)** | 200 | Specificity checks — "without" in idioms, etc. |

### Distribution Across Slices (Main Set)

| Slice | % of Main Set | Purpose |
|-------|---------------|---------|
| MinPairs | 20% | Mechanistic probe set |
| ExplicitMention | 40% | Core evaluation |
| Omission | 40% | Realistic stress test |

---

## 7. Controls

### Non-Flip Controls
Transformations where ranking **should NOT change**:
- Punctuation changes
- Casing changes
- Synonym substitution (non-semantic)
- Word order permutation (where meaning is preserved)

Purpose: Verify the model doesn't break on trivial variations.

### Adversarial Controls
Examples where "without" / "not" appear but **are NOT exclusion constraints**:
- "without further ado"
- "without doubt"
- "not only X but also Y"
- "cannot be overstated"

Purpose: Verify any "negation circuit" we find isn't just a keyword detector.

---

## 8. Success Criteria

### For Dataset Quality
1. **Inter-annotator agreement** on gold set > 90%
2. **Topical similarity** between doc_pos and doc_neg is verifiable
3. **Constraint validity** — doc_neg actually violates, doc_pos actually satisfies

### For Baseline Evaluation
1. **Non-trivial accuracy** — baseline rerankers achieve >50% but <100% pairwise accuracy (if 100%, dataset is too easy; if 50%, model is random)
2. **Measurable score gaps** — Δ values are non-zero and consistent
3. **Slice differentiation** — MinPairs should show cleaner results than Omission

### For Mechanistic Analysis (downstream)
1. Baseline results provide **behavioral signal** for patching/ablation
2. Some examples show **failure cases** (where model prefers doc_neg) — these are valuable for intervention analysis

---

## 9. References

### Key Papers
- **NevIR**: Negation in Neural Information Retrieval ([Weller et al., 2024](https://arxiv.org/abs/2305.07614)) — Benchmark with doc pairs differing only by negation
- **BEIR**: Benchmarking IR ([Thakur et al., 2021](https://arxiv.org/abs/2104.08663)) — Multi-domain evaluation
- **Activation Patching Best Practices** ([Kramár et al., 2024](https://arxiv.org/abs/2404.15255)) — Methodological guide for causal interventions

### Tools
- **Pyserini**: BM25 retrieval with prebuilt indexes ([GitHub](https://github.com/castorini/pyserini))
- **BEIR Library**: Dataset loading ([GitHub](https://github.com/beir-cellar/beir))
- **ir_datasets**: Unified IR dataset interface ([ir-datasets.com](https://ir-datasets.com))
- **Sentence-Transformers**: CrossEncoder API ([sbert.net](https://www.sbert.net/docs/package_reference/cross_encoder/cross_encoder.html))

### Models
- `cross-encoder/ms-marco-MiniLM-L6-v2` — Fast baseline reranker
- `BAAI/bge-reranker-base` — Stronger reranker

---

## 10. Quick Start

```bash
# Install dependencies
pip install -e ".[dev,notebooks]"

# Download MS MARCO index (one-time, ~2GB)
python scripts/01_download_data.py

# Run full pipeline
./scripts/run_pipeline.sh configs/negation_v0.yaml

# Or run stages individually
python scripts/03_generate_queries.py --config configs/negation_v0.yaml
python scripts/04_retrieve_candidates.py --config configs/negation_v0.yaml
python scripts/05_mine_pairs.py --config configs/negation_v0.yaml
python scripts/08_eval_baselines.py --config configs/negation_v0.yaml
```

---

## 11. FAQ

**Q: Why not just use NevIR directly?**
A: NevIR is great for benchmarking, but we need control over construction for mechanistic analysis. Our MinPairs slice is NevIR-inspired, but we also need ExplicitMention and Omission slices for different scientific questions.

**Q: Why MS MARCO and not BEIR?**
A: MS MARCO has a prebuilt Pyserini index and is well-understood. We'll add BEIR domains later for drift experiments across domains.

**Q: How do we handle polysemy (e.g., "Selenium" the element vs the tool)?**
A: We use surface form matching as a heuristic and rely on topical context (web scraping queries will surface the tool, not the element). Manual gold verification catches edge cases.

**Q: What if the baseline reranker is already perfect on negation?**
A: That's informative! It means the model has robust negation handling, and we can look for where that capability lives. More likely, we'll see partial success with interesting failure modes.
