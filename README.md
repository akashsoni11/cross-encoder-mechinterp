# ConstraintSuite

A diagnostic dataset for evaluating constraint-sensitive relevance in cross-encoder rerankers, designed to support mechanistic interpretability research.

## Overview

ConstraintSuite tests whether cross-encoder rerankers can handle **negation constraints** in queries. Each example is a pairwise ranking test where the model must score a constraint-satisfying document higher than a constraint-violating one.

```
Query: "python web scraping without selenium"

doc_pos: BeautifulSoup tutorial (satisfies constraint)
doc_neg: Selenium WebDriver guide (violates constraint)

✓ Correct if: score(doc_pos) > score(doc_neg)
```

### Research Goal

This dataset supports research into:
- **Mechanistic interpretability**: Do rerankers have identifiable "negation circuits"?
- **Causal localization**: Can we find these circuits via activation patching/ablation?
- **Mechanistic drift**: Do these circuits change during sequential fine-tuning?

## Installation

```bash
# Clone the repository
git clone https://github.com/tanujsharma/constraintsuite.git
cd constraintsuite

# Install with dependencies
pip install -e ".[dev,notebooks]"

# Download MS MARCO index (~2GB, one-time)
python scripts/01_download_data.py
```

### Requirements

- Python 3.10+
- ~2GB disk space for BM25 index
- GPU recommended for reranker inference

## Quick Start

```bash
# Run full pipeline
./scripts/run_pipeline.sh configs/negation_v0.yaml

# Or run stages individually
python scripts/03_generate_queries.py --config configs/negation_v0.yaml
python scripts/04_retrieve_candidates.py --config configs/negation_v0.yaml
python scripts/05_mine_pairs.py --config configs/negation_v0.yaml
python scripts/08_eval_baselines.py --config configs/negation_v0.yaml
```

## Dataset Structure

### Three Slices

| Slice | Description | Purpose |
|-------|-------------|---------|
| **MinPairs** | Near-identical docs, single edit | Cleanest for mechanistic analysis |
| **ExplicitMention** | doc_pos: "peanut-free", doc_neg: "contains peanuts" | Balanced lexical pressure |
| **Omission** | doc_pos never mentions Y | Real-world stress test |

### Data Format (JSONL)

```json
{
  "id": "negation_explicit_msmarco_000123",
  "suite": "negation_explicit",
  "query": {
    "base": "python web scraping selenium",
    "neg": "python web scraping without selenium"
  },
  "constraint": {
    "type": "exclude",
    "y": "selenium"
  },
  "docs": {
    "pos": {"text": "Use requests + BeautifulSoup..."},
    "neg": {"text": "Selenium WebDriver tutorial..."}
  },
  "tags": {
    "difficulty": "medium",
    "lexical_overlap_bin": "high"
  }
}
```

### Dataset Sizes

| Set | Target Size | Purpose |
|-----|-------------|---------|
| Main | 1k-5k | Evaluation |
| Gold | 30-60 | Manual verification |
| Controls | 500-700 | Sanity checks |

## Project Structure

```
constraintsuite/
├── configs/                    # Configuration files
│   └── negation_v0.yaml
├── data/
│   ├── raw/                    # Downloaded corpora (gitignored)
│   ├── intermediate/           # Processing artifacts
│   └── release/                # Final datasets
│       └── negation_v0/
├── src/constraintsuite/        # Core library
│   ├── data_loading.py
│   ├── retrieval.py
│   ├── query_generation.py
│   ├── pair_mining.py
│   ├── filtering.py
│   ├── tagging.py
│   └── evaluation.py
├── scripts/                    # Pipeline scripts
│   ├── 01_download_data.py
│   ├── ...
│   └── run_pipeline.sh
├── notebooks/                  # Analysis notebooks
└── tests/                      # Unit tests
```

## Documentation

See [DATASET_CONSTRUCTION.md](DATASET_CONSTRUCTION.md) for:
- Detailed construction methodology
- Code examples for each pipeline stage
- Full data schema
- References and related work

## Evaluation

```python
from constraintsuite.evaluation import evaluate_dataset

results = evaluate_dataset(
    "cross-encoder/ms-marco-MiniLM-L6-v2",
    examples
)
print(f"Accuracy: {results.pairwise_accuracy:.2%}")
print(f"Mean Δ: {results.mean_score_gap:.4f}")
```

## Models

Supported rerankers:
- `cross-encoder/ms-marco-MiniLM-L6-v2` (fast baseline)
- `BAAI/bge-reranker-base` (stronger)

## References

- **NevIR**: Negation in Neural IR ([Weller et al., 2024](https://arxiv.org/abs/2305.07614))
- **BEIR**: Benchmarking IR ([Thakur et al., 2021](https://arxiv.org/abs/2104.08663))
- **Activation Patching**: Best practices ([Kramár et al., 2024](https://arxiv.org/abs/2404.15255))

## License

MIT License
