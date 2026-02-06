"""
ConstraintSuite: Dataset for evaluating constraint-sensitive relevance in cross-encoder rerankers.

This package provides tools for:
- Generating constraint-based query-document pairs
- Mining positive/negative document pairs
- Evaluating reranker performance on constraint handling
- Supporting mechanistic interpretability research

Example usage:
    from constraintsuite import load_config, generate_dataset, evaluate_reranker

    config = load_config("configs/negation_v0.yaml")
    dataset = generate_dataset(config)
    results = evaluate_reranker("cross-encoder/ms-marco-MiniLM-L6-v2", dataset)
"""

__version__ = "0.1.0"
__author__ = "Tanuj Sharma"

# Core utilities
from constraintsuite.utils import (
    load_config,
    save_jsonl,
    load_jsonl,
    iter_jsonl,
    setup_logging,
    set_seed,
    ensure_dir,
    get_logger,
)

# Data loading
from constraintsuite.data_loading import (
    load_msmarco_queries,
    load_msmarco_corpus,
    load_beir_dataset,
    iter_queries,
    get_query_entities,
)

# Query generation
from constraintsuite.query_generation import (
    generate_negated_query,
    batch_generate_queries,
    expand_surface_forms,
    GeneratedQuery,
)

# Pair mining
from constraintsuite.pair_mining import (
    mine_pair,
    mine_minpair,
    contains_y,
    y_is_negated_nearby,
    classify_pair_slice,
    MinedPair,
)

# Filtering
from constraintsuite.filtering import (
    filter_pair,
    batch_filter,
    FilterResult,
)

# Tagging
from constraintsuite.tagging import (
    tag_pair,
    batch_tag,
    sample_gold_set,
    compute_distribution_stats,
    PairTags,
)

# Lazy imports for modules with heavy external dependencies (Java/Pyserini, CrossEncoder, Codex CLI)
def __getattr__(name):
    if name == "BM25Retriever":
        from constraintsuite.retrieval import BM25Retriever
        return BM25Retriever
    if name in ("CrossEncoderScorer", "evaluate_dataset", "evaluate_example", "EvaluationResult"):
        import constraintsuite.evaluation as _eval
        return getattr(_eval, name)
    if name in (
        "run_codex_agent", "fix_minpair_grammar", "validate_gold_example",
        "expand_surface_forms_llm", "classify_ambiguous_pair",
        "batch_validate_gold_set", "batch_fix_grammar",
    ):
        import constraintsuite.llm_utils as _llm
        return getattr(_llm, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # Version
    "__version__",
    # Utils
    "load_config",
    "save_jsonl",
    "load_jsonl",
    "iter_jsonl",
    "setup_logging",
    "set_seed",
    "ensure_dir",
    "get_logger",
    # Data loading
    "load_msmarco_queries",
    "load_msmarco_corpus",
    "load_beir_dataset",
    "iter_queries",
    "get_query_entities",
    # Query generation
    "generate_negated_query",
    "batch_generate_queries",
    "expand_surface_forms",
    "GeneratedQuery",
    # Retrieval
    "BM25Retriever",
    # Pair mining
    "mine_pair",
    "mine_minpair",
    "contains_y",
    "y_is_negated_nearby",
    "classify_pair_slice",
    "MinedPair",
    # Filtering
    "filter_pair",
    "batch_filter",
    "FilterResult",
    # Tagging
    "tag_pair",
    "batch_tag",
    "sample_gold_set",
    "compute_distribution_stats",
    "PairTags",
    # Evaluation
    "CrossEncoderScorer",
    "evaluate_dataset",
    "evaluate_example",
    "EvaluationResult",
    # LLM utilities (via Codex CLI)
    "run_codex_agent",
    "fix_minpair_grammar",
    "validate_gold_example",
    "expand_surface_forms_llm",
    "classify_ambiguous_pair",
    "batch_validate_gold_set",
    "batch_fix_grammar",
]
