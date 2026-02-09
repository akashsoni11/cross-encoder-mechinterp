"""
Activation patching infrastructure for mechanistic interpretability.

This module provides tools for causal localization of negation processing
in cross-encoder rerankers using activation patching and ablation.

Target model: cross-encoder/ms-marco-MiniLM-L6-v2
  - 6 layers, 12 heads, 384 hidden dim, ~22M params

Approach:
  - Clean input: (q_neg, d_pos) — correct pair
  - Corrupt input: (q_neg, d_neg) — wrong doc (doc-swap corruption)
  - Patch one site from clean into corrupt → measure effect on output
  - Normalized effect = (patched - corrupt) / (clean - corrupt)

Usage:
    from constraintsuite.patching import PatchableModel, run_component_scan

    model = PatchableModel("cross-encoder/ms-marco-MiniLM-L6-v2", device="mps")
    results = run_component_scan(model, examples)
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger("constraintsuite")

# ---------------------------------------------------------------------------
# Constants for ms-marco-MiniLM-L6-v2
# ---------------------------------------------------------------------------
NUM_LAYERS = 6
NUM_HEADS = 12
HEAD_DIM = 32  # 384 / 12
HIDDEN_DIM = 384
FFN_DIM = 1536  # 4 * 384


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class PatchingSite:
    """Identifies a hookable location in the model."""

    name: str  # e.g. "layer.3.attn_out" or "layer.2.head.7"
    module_path: str  # e.g. "bert.encoder.layer.3.attention.output"
    layer_idx: int | None = None
    head_idx: int | None = None

    @property
    def is_head_level(self) -> bool:
        return self.head_idx is not None


@dataclass
class PatchingResult:
    """Per-example patching result for one site."""

    example_id: str
    site_name: str
    clean_score: float
    corrupt_score: float
    patched_score: float
    total_effect: float  # clean - corrupt
    patched_effect: float  # patched - corrupt
    normalized_effect: float  # patched_effect / total_effect


@dataclass
class ScanResult:
    """Aggregated patching scan results."""

    scan_type: str  # "component" or "head"
    model_name: str
    num_examples: int
    sites: list[str]
    per_example: list[PatchingResult] = field(default_factory=list)
    # Aggregated stats per site: {site_name: {mean, std, median, count}}
    site_stats: dict[str, dict[str, float]] = field(default_factory=dict)
    # Per-slice breakdown: {slice_type: {site_name: {mean, std, count}}}
    slice_stats: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)
    elapsed_seconds: float = 0.0

    def compute_stats(self) -> None:
        """Compute aggregated statistics from per-example results."""
        from collections import defaultdict

        by_site: dict[str, list[float]] = defaultdict(list)
        by_slice_site: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )

        for r in self.per_example:
            by_site[r.site_name].append(r.normalized_effect)

        for site_name, effects in by_site.items():
            arr = np.array(effects)
            self.site_stats[site_name] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "median": float(np.median(arr)),
                "abs_mean": float(np.mean(np.abs(arr))),
                "count": len(arr),
            }

    def to_dict(self) -> dict[str, Any]:
        return {
            "scan_type": self.scan_type,
            "model_name": self.model_name,
            "num_examples": self.num_examples,
            "sites": self.sites,
            "site_stats": self.site_stats,
            "slice_stats": self.slice_stats,
            "elapsed_seconds": self.elapsed_seconds,
            "per_example": [
                {
                    "example_id": r.example_id,
                    "site_name": r.site_name,
                    "clean_score": r.clean_score,
                    "corrupt_score": r.corrupt_score,
                    "patched_score": r.patched_score,
                    "total_effect": r.total_effect,
                    "patched_effect": r.patched_effect,
                    "normalized_effect": r.normalized_effect,
                }
                for r in self.per_example
            ],
        }


@dataclass
class AblationResult:
    """Results from an ablation study."""

    model_name: str
    method: str  # "zero" or "mean"
    num_examples: int
    # {site_name: {accuracy, mean_score_gap, accuracy_drop}}
    targeted_results: dict[str, dict[str, float]] = field(default_factory=dict)
    # [{sites: [...], accuracy, mean_score_gap}]
    random_controls: list[dict[str, Any]] = field(default_factory=list)
    baseline_accuracy: float = 0.0
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "method": self.method,
            "num_examples": self.num_examples,
            "baseline_accuracy": self.baseline_accuracy,
            "targeted_results": self.targeted_results,
            "random_controls": self.random_controls,
            "elapsed_seconds": self.elapsed_seconds,
        }


# ---------------------------------------------------------------------------
# Site enumeration
# ---------------------------------------------------------------------------
def get_component_sites() -> list[PatchingSite]:
    """Return the 25 component-level patching sites."""
    sites = [
        PatchingSite(
            name="embed",
            module_path="bert.embeddings",
        ),
    ]
    for i in range(NUM_LAYERS):
        sites.extend([
            PatchingSite(
                name=f"layer.{i}.attn_self",
                module_path=f"bert.encoder.layer.{i}.attention.self",
                layer_idx=i,
            ),
            PatchingSite(
                name=f"layer.{i}.attn_out",
                module_path=f"bert.encoder.layer.{i}.attention.output",
                layer_idx=i,
            ),
            PatchingSite(
                name=f"layer.{i}.ffn_mid",
                module_path=f"bert.encoder.layer.{i}.intermediate",
                layer_idx=i,
            ),
            PatchingSite(
                name=f"layer.{i}.ffn_out",
                module_path=f"bert.encoder.layer.{i}.output",
                layer_idx=i,
            ),
        ])
    sites.append(
        PatchingSite(
            name="pooler",
            module_path="bert.pooler",
        )
    )
    return sites


def get_head_sites(layers: list[int] | None = None) -> list[PatchingSite]:
    """Return head-level patching sites (slices of attention self output)."""
    if layers is None:
        layers = list(range(NUM_LAYERS))
    sites = []
    for i in layers:
        for h in range(NUM_HEADS):
            sites.append(
                PatchingSite(
                    name=f"layer.{i}.head.{h}",
                    module_path=f"bert.encoder.layer.{i}.attention.self",
                    layer_idx=i,
                    head_idx=h,
                )
            )
    return sites


# ---------------------------------------------------------------------------
# PatchableModel
# ---------------------------------------------------------------------------
class PatchableModel:
    """
    Wraps a HuggingFace cross-encoder model with hook-based activation patching.

    Uses raw PyTorch forward hooks for maximum control and no extra dependencies.
    Caches activations on CPU to prevent MPS/CUDA memory exhaustion.

    Example:
        >>> model = PatchableModel("cross-encoder/ms-marco-MiniLM-L6-v2", device="mps")
        >>> inputs = model.tokenize_pair("python without selenium", "BeautifulSoup tutorial")
        >>> score = model.score(inputs)
        >>> cache = model.cache_activations(inputs, get_component_sites())
        >>> patched = model.patch_and_score(corrupt_inputs, cache, site)
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2",
        device: str = "mps",
        max_length: int = 256,
    ):
        self.model_name = model_name
        self.max_length = max_length

        # Resolve device
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)

        logger.info(f"Loading model: {model_name} on {device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # Build module lookup for fast hook registration
        self._module_lookup: dict[str, torch.nn.Module] = {}
        for name, module in self.model.named_modules():
            self._module_lookup[name] = module

    def _get_module(self, module_path: str) -> torch.nn.Module:
        """Get a module by its dot-separated path."""
        if module_path not in self._module_lookup:
            raise ValueError(f"Module not found: {module_path}")
        return self._module_lookup[module_path]

    def tokenize_pair(self, query: str, document: str) -> dict[str, torch.Tensor]:
        """
        Tokenize a query-document pair for cross-encoder input.

        Uses fixed padding to max_length for shape compatibility between
        clean and corrupt tensors.
        """
        inputs = self.tokenizer(
            query,
            document,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in inputs.items()}

    @torch.no_grad()
    def score(self, inputs: dict[str, torch.Tensor]) -> float:
        """Run a forward pass and return the scalar relevance logit."""
        outputs = self.model(**inputs)
        return outputs.logits.squeeze(-1).item()

    @torch.no_grad()
    def cache_activations(
        self,
        inputs: dict[str, torch.Tensor],
        sites: list[PatchingSite],
    ) -> dict[str, torch.Tensor]:
        """
        Run a forward pass and collect activations at specified sites.

        Activations are stored on CPU to prevent device memory exhaustion.

        Returns:
            Dict mapping site name -> activation tensor (on CPU).
        """
        cache: dict[str, torch.Tensor] = {}
        handles = []

        for site in sites:
            module = self._get_module(site.module_path)

            # Use default arg binding to capture site in closure
            def make_hook(site_name: str):
                def hook_fn(module, input, output):
                    # output can be a tuple (e.g., attention self returns tuple)
                    if isinstance(output, tuple):
                        tensor = output[0]
                    else:
                        tensor = output
                    cache[site_name] = tensor.detach().cpu()
                return hook_fn

            handle = module.register_forward_hook(make_hook(site.name))
            handles.append(handle)

        # Forward pass
        self.model(**inputs)

        # Clean up hooks
        for handle in handles:
            handle.remove()

        return cache

    @torch.no_grad()
    def patch_and_score(
        self,
        corrupt_inputs: dict[str, torch.Tensor],
        clean_cache: dict[str, torch.Tensor],
        target_site: PatchingSite,
    ) -> float:
        """
        Run corrupt input through the model, patching one site from clean cache.

        For component-level sites, the entire activation is replaced.
        For head-level sites, only the head's slice is replaced.

        Returns:
            Model score with the patched activation.
        """
        site_name = target_site.name
        if site_name not in clean_cache:
            raise ValueError(f"Site {site_name} not found in clean cache")

        clean_act = clean_cache[site_name].to(self.device)
        module = self._get_module(target_site.module_path)

        def patch_hook(mod, input, output):
            if isinstance(output, tuple):
                tensor = output[0]
                is_tuple = True
            else:
                tensor = output
                is_tuple = False

            if target_site.is_head_level:
                # Patch only the head's slice: [h*HEAD_DIM : (h+1)*HEAD_DIM]
                h = target_site.head_idx
                start = h * HEAD_DIM
                end = (h + 1) * HEAD_DIM
                patched = tensor.clone()
                patched[:, :, start:end] = clean_act[:, :, start:end]
            else:
                patched = clean_act

            if is_tuple:
                return (patched,) + output[1:]
            return patched

        handle = module.register_forward_hook(patch_hook)
        try:
            outputs = self.model(**corrupt_inputs)
            score = outputs.logits.squeeze(-1).item()
        finally:
            handle.remove()

        return score

    @torch.no_grad()
    def ablate_and_score(
        self,
        inputs: dict[str, torch.Tensor],
        target_sites: list[PatchingSite],
        method: Literal["zero", "mean"] = "zero",
        mean_cache: dict[str, torch.Tensor] | None = None,
    ) -> float:
        """
        Run input through model with specified sites ablated.

        Args:
            inputs: Tokenized input.
            target_sites: Sites to ablate.
            method: "zero" for zero ablation, "mean" for mean ablation.
            mean_cache: Pre-computed mean activations for mean ablation.

        Returns:
            Model score with ablated activations.
        """
        handles = []

        for site in target_sites:
            module = self._get_module(site.module_path)

            def make_ablation_hook(s: PatchingSite):
                def hook_fn(mod, input, output):
                    if isinstance(output, tuple):
                        tensor = output[0]
                        is_tuple = True
                    else:
                        tensor = output
                        is_tuple = False

                    if method == "zero":
                        if s.is_head_level:
                            ablated = tensor.clone()
                            h = s.head_idx
                            start = h * HEAD_DIM
                            end = (h + 1) * HEAD_DIM
                            ablated[:, :, start:end] = 0.0
                        else:
                            ablated = torch.zeros_like(tensor)
                    elif method == "mean":
                        if mean_cache is None or s.name not in mean_cache:
                            raise ValueError(
                                f"Mean cache required for mean ablation of {s.name}"
                            )
                        mean_act = mean_cache[s.name].to(tensor.device)
                        if s.is_head_level:
                            ablated = tensor.clone()
                            h = s.head_idx
                            start = h * HEAD_DIM
                            end = (h + 1) * HEAD_DIM
                            ablated[:, :, start:end] = mean_act[:, :, start:end]
                        else:
                            ablated = mean_act.expand_as(tensor)
                    else:
                        raise ValueError(f"Unknown ablation method: {method}")

                    if is_tuple:
                        return (ablated,) + output[1:]
                    return ablated

                return hook_fn

            handle = module.register_forward_hook(make_ablation_hook(site))
            handles.append(handle)

        try:
            outputs = self.model(**inputs)
            score = outputs.logits.squeeze(-1).item()
        finally:
            for handle in handles:
                handle.remove()

        return score


# ---------------------------------------------------------------------------
# Example extraction helper
# ---------------------------------------------------------------------------
def _extract_patching_fields(example: dict[str, Any]) -> dict[str, Any]:
    """Extract fields needed for patching from a dataset example."""
    query_info = example.get("query", {})
    if isinstance(query_info, dict):
        q_neg = query_info.get("neg", query_info.get("negated", ""))
    else:
        q_neg = str(query_info)

    doc_pos = example.get("doc_pos", example.get("docs", {}).get("pos", {}))
    doc_neg = example.get("doc_neg", example.get("docs", {}).get("neg", {}))

    text_pos = doc_pos.get("text", "") if isinstance(doc_pos, dict) else str(doc_pos)
    text_neg = doc_neg.get("text", "") if isinstance(doc_neg, dict) else str(doc_neg)

    return {
        "id": example.get("id", ""),
        "q_neg": q_neg,
        "text_pos": text_pos,
        "text_neg": text_neg,
        "slice_type": example.get("slice_type", "unknown"),
    }


# ---------------------------------------------------------------------------
# Checkpointing helpers
# ---------------------------------------------------------------------------
def _save_checkpoint(
    results: list[PatchingResult],
    checkpoint_dir: Path,
    prefix: str,
    example_idx: int,
) -> None:
    """Save intermediate results to a checkpoint file."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"checkpoint_{prefix}_{example_idx:06d}.jsonl"
    with open(path, "w") as f:
        for r in results:
            line = {
                "example_id": r.example_id,
                "site_name": r.site_name,
                "clean_score": r.clean_score,
                "corrupt_score": r.corrupt_score,
                "patched_score": r.patched_score,
                "total_effect": r.total_effect,
                "patched_effect": r.patched_effect,
                "normalized_effect": r.normalized_effect,
            }
            f.write(json.dumps(line) + "\n")
    logger.debug(f"Checkpoint saved: {path}")


def _load_checkpoint(checkpoint_dir: Path, prefix: str) -> list[PatchingResult]:
    """Load the latest checkpoint if available."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return []

    checkpoints = sorted(checkpoint_dir.glob(f"checkpoint_{prefix}_*.jsonl"))
    if not checkpoints:
        return []

    latest = checkpoints[-1]
    logger.info(f"Resuming from checkpoint: {latest}")
    results = []
    with open(latest) as f:
        for line in f:
            data = json.loads(line.strip())
            results.append(PatchingResult(**data))
    return results


# ---------------------------------------------------------------------------
# Component scan
# ---------------------------------------------------------------------------
def run_component_scan(
    model: PatchableModel,
    examples: list[dict[str, Any]],
    checkpoint_dir: str | Path | None = None,
    checkpoint_every: int = 100,
    min_total_effect: float = 0.5,
    resume: bool = False,
) -> ScanResult:
    """
    Scan all 25 component-level sites across the dataset.

    For each example:
      1. Compute clean score (q_neg, d_pos) and corrupt score (q_neg, d_neg)
      2. Cache clean activations at all sites
      3. For each site, patch clean activation into corrupt run
      4. Compute normalized effect

    Args:
        model: PatchableModel instance.
        examples: Dataset examples.
        checkpoint_dir: Directory for checkpoint files.
        checkpoint_every: Checkpoint frequency (examples).
        min_total_effect: Minimum |clean - corrupt| to include example.
        resume: Whether to resume from the latest checkpoint.

    Returns:
        ScanResult with per-example and aggregated statistics.
    """
    sites = get_component_sites()
    site_names = [s.name for s in sites]

    # Handle resume
    existing_results: list[PatchingResult] = []
    start_idx = 0
    if resume and checkpoint_dir:
        existing_results = _load_checkpoint(Path(checkpoint_dir), "component")
        if existing_results:
            seen_ids = {r.example_id for r in existing_results}
            # Find first example not yet processed
            for i, ex in enumerate(examples):
                ex_id = ex.get("id", f"example_{i}")
                if ex_id not in seen_ids:
                    start_idx = i
                    break
            else:
                start_idx = len(examples)
            logger.info(
                f"Resuming from example {start_idx} "
                f"({len(existing_results)} results loaded)"
            )

    all_results = list(existing_results)
    skipped = 0
    t0 = time.time()

    for idx in tqdm(range(start_idx, len(examples)), desc="Component scan"):
        example = examples[idx]
        fields = _extract_patching_fields(example)
        example_id = fields["id"] or f"example_{idx}"

        # Tokenize clean and corrupt
        clean_inputs = model.tokenize_pair(fields["q_neg"], fields["text_pos"])
        corrupt_inputs = model.tokenize_pair(fields["q_neg"], fields["text_neg"])

        # Get clean and corrupt scores
        clean_score = model.score(clean_inputs)
        corrupt_score = model.score(corrupt_inputs)
        total_effect = clean_score - corrupt_score

        # Filter low-signal examples
        if abs(total_effect) < min_total_effect:
            skipped += 1
            continue

        # Cache clean activations
        clean_cache = model.cache_activations(clean_inputs, sites)

        # Patch each site
        for site in sites:
            patched_score = model.patch_and_score(corrupt_inputs, clean_cache, site)
            patched_effect = patched_score - corrupt_score
            normalized = patched_effect / total_effect

            all_results.append(
                PatchingResult(
                    example_id=example_id,
                    site_name=site.name,
                    clean_score=clean_score,
                    corrupt_score=corrupt_score,
                    patched_score=patched_score,
                    total_effect=total_effect,
                    patched_effect=patched_effect,
                    normalized_effect=normalized,
                )
            )

        # Checkpoint
        if checkpoint_dir and (idx + 1) % checkpoint_every == 0:
            _save_checkpoint(all_results, Path(checkpoint_dir), "component", idx)

    elapsed = time.time() - t0

    # Final checkpoint
    if checkpoint_dir:
        _save_checkpoint(all_results, Path(checkpoint_dir), "component", len(examples))

    logger.info(
        f"Component scan complete: {len(examples)} examples, "
        f"{skipped} skipped (low signal), {elapsed:.1f}s"
    )

    scan = ScanResult(
        scan_type="component",
        model_name=model.model_name,
        num_examples=len(examples) - skipped,
        sites=site_names,
        per_example=all_results,
        elapsed_seconds=elapsed,
    )
    scan.compute_stats()
    return scan


# ---------------------------------------------------------------------------
# Head scan
# ---------------------------------------------------------------------------
def run_head_scan(
    model: PatchableModel,
    examples: list[dict[str, Any]],
    layers: list[int] | None = None,
    checkpoint_dir: str | Path | None = None,
    checkpoint_every: int = 100,
    min_total_effect: float = 0.5,
    resume: bool = False,
) -> ScanResult:
    """
    Scan head-level sites (attention head slices) across the dataset.

    Same counterfactual design as component scan but patches individual
    attention heads via slicing.

    Args:
        model: PatchableModel instance.
        examples: Dataset examples.
        layers: Which layers to scan (default: all 6).
        checkpoint_dir: Directory for checkpoint files.
        checkpoint_every: Checkpoint frequency.
        min_total_effect: Minimum |total_effect| to include example.
        resume: Whether to resume from latest checkpoint.

    Returns:
        ScanResult with per-example and aggregated statistics.
    """
    sites = get_head_sites(layers)
    site_names = [s.name for s in sites]

    # We also need the component-level attn_self sites for caching
    # (head sites are slices of attn_self output)
    cache_layers = layers if layers is not None else list(range(NUM_LAYERS))
    cache_sites = [
        PatchingSite(
            name=f"layer.{i}.attn_self",
            module_path=f"bert.encoder.layer.{i}.attention.self",
            layer_idx=i,
        )
        for i in cache_layers
    ]

    # Handle resume
    existing_results: list[PatchingResult] = []
    start_idx = 0
    if resume and checkpoint_dir:
        existing_results = _load_checkpoint(Path(checkpoint_dir), "head")
        if existing_results:
            seen_ids = {r.example_id for r in existing_results}
            for i, ex in enumerate(examples):
                ex_id = ex.get("id", f"example_{i}")
                if ex_id not in seen_ids:
                    start_idx = i
                    break
            else:
                start_idx = len(examples)

    all_results = list(existing_results)
    skipped = 0
    t0 = time.time()

    for idx in tqdm(range(start_idx, len(examples)), desc="Head scan"):
        example = examples[idx]
        fields = _extract_patching_fields(example)
        example_id = fields["id"] or f"example_{idx}"

        clean_inputs = model.tokenize_pair(fields["q_neg"], fields["text_pos"])
        corrupt_inputs = model.tokenize_pair(fields["q_neg"], fields["text_neg"])

        clean_score = model.score(clean_inputs)
        corrupt_score = model.score(corrupt_inputs)
        total_effect = clean_score - corrupt_score

        if abs(total_effect) < min_total_effect:
            skipped += 1
            continue

        # Cache clean activations at attn_self level
        clean_cache = model.cache_activations(clean_inputs, cache_sites)

        # Patch each head
        for site in sites:
            # Head sites use the same module as attn_self, just patch a slice
            cache_key = f"layer.{site.layer_idx}.attn_self"
            head_cache = {site.name: clean_cache[cache_key]}
            patched_score = model.patch_and_score(corrupt_inputs, head_cache, site)
            patched_effect = patched_score - corrupt_score
            normalized = patched_effect / total_effect

            all_results.append(
                PatchingResult(
                    example_id=example_id,
                    site_name=site.name,
                    clean_score=clean_score,
                    corrupt_score=corrupt_score,
                    patched_score=patched_score,
                    total_effect=total_effect,
                    patched_effect=patched_effect,
                    normalized_effect=normalized,
                )
            )

        if checkpoint_dir and (idx + 1) % checkpoint_every == 0:
            _save_checkpoint(all_results, Path(checkpoint_dir), "head", idx)

    elapsed = time.time() - t0

    if checkpoint_dir:
        _save_checkpoint(all_results, Path(checkpoint_dir), "head", len(examples))

    logger.info(
        f"Head scan complete: {len(examples)} examples, "
        f"{skipped} skipped, {elapsed:.1f}s"
    )

    scan = ScanResult(
        scan_type="head",
        model_name=model.model_name,
        num_examples=len(examples) - skipped,
        sites=site_names,
        per_example=all_results,
        elapsed_seconds=elapsed,
    )
    scan.compute_stats()
    return scan


# ---------------------------------------------------------------------------
# Ablation study
# ---------------------------------------------------------------------------
def run_ablation_study(
    model: PatchableModel,
    examples: list[dict[str, Any]],
    target_sites: list[PatchingSite],
    method: Literal["zero", "mean"] = "zero",
    n_random_controls: int = 5,
    mean_cache: dict[str, torch.Tensor] | None = None,
    random_seed: int = 42,
) -> AblationResult:
    """
    Ablation study: ablate target sites and compare to random controls.

    For each ablation configuration:
      1. Run all examples with sites ablated
      2. Compute pairwise accuracy and mean score gap

    Args:
        model: PatchableModel instance.
        examples: Dataset examples.
        target_sites: Sites identified as important ("hot") from scan.
        method: Ablation method ("zero" or "mean").
        n_random_controls: Number of random ablation controls to run.
        mean_cache: Pre-computed mean activations for mean ablation.
        random_seed: Seed for random control selection.

    Returns:
        AblationResult with targeted vs random comparison.
    """
    rng = np.random.RandomState(random_seed)
    t0 = time.time()

    # First compute baseline (no ablation)
    logger.info("Computing baseline (no ablation)...")
    baseline_correct = 0
    baseline_gaps = []
    for example in tqdm(examples, desc="Baseline"):
        fields = _extract_patching_fields(example)
        clean_inputs = model.tokenize_pair(fields["q_neg"], fields["text_pos"])
        corrupt_inputs = model.tokenize_pair(fields["q_neg"], fields["text_neg"])
        score_pos = model.score(clean_inputs)
        score_neg = model.score(corrupt_inputs)
        if score_pos > score_neg:
            baseline_correct += 1
        baseline_gaps.append(score_pos - score_neg)

    baseline_accuracy = baseline_correct / len(examples) if examples else 0.0
    baseline_mean_gap = float(np.mean(baseline_gaps)) if baseline_gaps else 0.0
    logger.info(f"Baseline: accuracy={baseline_accuracy:.2%}, gap={baseline_mean_gap:.4f}")

    # Targeted ablation (each site individually)
    targeted_results = {}
    for site in tqdm(target_sites, desc="Targeted ablation"):
        correct = 0
        gaps = []
        for example in examples:
            fields = _extract_patching_fields(example)
            clean_inputs = model.tokenize_pair(fields["q_neg"], fields["text_pos"])
            corrupt_inputs = model.tokenize_pair(fields["q_neg"], fields["text_neg"])
            score_pos = model.ablate_and_score(
                clean_inputs, [site], method=method, mean_cache=mean_cache
            )
            score_neg = model.ablate_and_score(
                corrupt_inputs, [site], method=method, mean_cache=mean_cache
            )
            if score_pos > score_neg:
                correct += 1
            gaps.append(score_pos - score_neg)

        accuracy = correct / len(examples) if examples else 0.0
        mean_gap = float(np.mean(gaps)) if gaps else 0.0
        targeted_results[site.name] = {
            "accuracy": accuracy,
            "mean_score_gap": mean_gap,
            "accuracy_drop": baseline_accuracy - accuracy,
        }
        logger.info(
            f"Ablated {site.name}: accuracy={accuracy:.2%} "
            f"(drop={baseline_accuracy - accuracy:+.2%})"
        )

    # Random controls: ablate same number of sites but randomly chosen
    all_component_sites = get_component_sites()
    target_count = len(target_sites)
    random_controls = []

    for ctrl_idx in range(n_random_controls):
        # Sample random sites (excluding target sites)
        non_target = [s for s in all_component_sites if s.name not in {t.name for t in target_sites}]
        if len(non_target) < target_count:
            non_target = all_component_sites  # fallback if target_count > available
        chosen_indices = rng.choice(len(non_target), size=min(target_count, len(non_target)), replace=False)
        chosen_sites = [non_target[i] for i in chosen_indices]

        correct = 0
        gaps = []
        for example in examples:
            fields = _extract_patching_fields(example)
            clean_inputs = model.tokenize_pair(fields["q_neg"], fields["text_pos"])
            corrupt_inputs = model.tokenize_pair(fields["q_neg"], fields["text_neg"])
            score_pos = model.ablate_and_score(
                clean_inputs, chosen_sites, method=method, mean_cache=mean_cache
            )
            score_neg = model.ablate_and_score(
                corrupt_inputs, chosen_sites, method=method, mean_cache=mean_cache
            )
            if score_pos > score_neg:
                correct += 1
            gaps.append(score_pos - score_neg)

        accuracy = correct / len(examples) if examples else 0.0
        mean_gap = float(np.mean(gaps)) if gaps else 0.0
        random_controls.append({
            "control_idx": ctrl_idx,
            "sites": [s.name for s in chosen_sites],
            "accuracy": accuracy,
            "mean_score_gap": mean_gap,
            "accuracy_drop": baseline_accuracy - accuracy,
        })
        logger.info(
            f"Random control {ctrl_idx}: accuracy={accuracy:.2%} "
            f"(drop={baseline_accuracy - accuracy:+.2%})"
        )

    elapsed = time.time() - t0

    return AblationResult(
        model_name=model.model_name,
        method=method,
        num_examples=len(examples),
        targeted_results=targeted_results,
        random_controls=random_controls,
        baseline_accuracy=baseline_accuracy,
        elapsed_seconds=elapsed,
    )


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------
def plot_component_heatmap(
    scan: ScanResult,
    output_path: str | Path | None = None,
    figsize: tuple[int, int] = (14, 5),
) -> Any:
    """
    Plot a layer x component heatmap of normalized patching effects.

    Returns the matplotlib figure.
    """
    import matplotlib.pyplot as plt

    components = ["attn_self", "attn_out", "ffn_mid", "ffn_out"]
    layers = list(range(NUM_LAYERS))

    # Build matrix
    matrix = np.zeros((len(layers), len(components)))
    for i, layer in enumerate(layers):
        for j, comp in enumerate(components):
            site_name = f"layer.{layer}.{comp}"
            if site_name in scan.site_stats:
                matrix[i, j] = scan.site_stats[site_name]["mean"]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix, cmap="RdBu_r", aspect="auto", vmin=-0.3, vmax=0.3)
    ax.set_xticks(range(len(components)))
    ax.set_xticklabels(components, rotation=45, ha="right")
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([f"Layer {i}" for i in layers])
    ax.set_xlabel("Component")
    ax.set_ylabel("Layer")
    ax.set_title(f"Activation Patching: Component Effects\n({scan.num_examples} examples)")

    # Add text annotations
    for i in range(len(layers)):
        for j in range(len(components)):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, label="Normalized Effect")
    fig.tight_layout()

    # Also add embed and pooler as sidebar text
    sidebar = []
    if "embed" in scan.site_stats:
        sidebar.append(f"embed: {scan.site_stats['embed']['mean']:.3f}")
    if "pooler" in scan.site_stats:
        sidebar.append(f"pooler: {scan.site_stats['pooler']['mean']:.3f}")
    if sidebar:
        ax.text(
            1.02, 0.5, "\n".join(sidebar),
            transform=ax.transAxes, fontsize=9, va="center",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Heatmap saved to {output_path}")

    return fig


def plot_head_heatmap(
    scan: ScanResult,
    output_path: str | Path | None = None,
    figsize: tuple[int, int] = (16, 5),
) -> Any:
    """
    Plot a layer x head heatmap of normalized patching effects.

    Returns the matplotlib figure.
    """
    import matplotlib.pyplot as plt

    layers = list(range(NUM_LAYERS))
    heads = list(range(NUM_HEADS))

    matrix = np.zeros((len(layers), len(heads)))
    for i, layer in enumerate(layers):
        for j, head in enumerate(heads):
            site_name = f"layer.{layer}.head.{head}"
            if site_name in scan.site_stats:
                matrix[i, j] = scan.site_stats[site_name]["mean"]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix, cmap="RdBu_r", aspect="auto", vmin=-0.3, vmax=0.3)
    ax.set_xticks(range(len(heads)))
    ax.set_xticklabels([f"H{h}" for h in heads])
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([f"Layer {i}" for i in layers])
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_title(f"Activation Patching: Head Effects\n({scan.num_examples} examples)")

    # Add text annotations
    for i in range(len(layers)):
        for j in range(len(heads)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7)

    fig.colorbar(im, ax=ax, label="Normalized Effect")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Heatmap saved to {output_path}")

    return fig


def get_hot_sites(
    scan: ScanResult,
    threshold: float = 0.1,
) -> list[tuple[str, float]]:
    """
    Identify "hot" sites with abs_mean effect above threshold.

    Returns list of (site_name, abs_mean_effect) sorted by effect size.
    """
    hot = []
    for site_name, stats in scan.site_stats.items():
        if stats["abs_mean"] >= threshold:
            hot.append((site_name, stats["abs_mean"]))
    hot.sort(key=lambda x: x[1], reverse=True)
    return hot
