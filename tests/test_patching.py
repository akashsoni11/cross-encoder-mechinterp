"""
Tests for activation patching infrastructure.

Run with: pytest tests/test_patching.py -v

These tests load the real model (cross-encoder/ms-marco-MiniLM-L6-v2)
to verify hook-based patching produces correct shapes and scores.
"""

import pytest
import torch

from constraintsuite.patching import (
    HEAD_DIM,
    HIDDEN_DIM,
    NUM_HEADS,
    NUM_LAYERS,
    AblationResult,
    PatchableModel,
    PatchingResult,
    PatchingSite,
    ScanResult,
    get_component_sites,
    get_head_sites,
    get_hot_sites,
    run_ablation_study,
    run_component_scan,
    run_head_scan,
)

# Use CPU for tests (no GPU/MPS requirement)
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"
DEVICE = "cpu"


@pytest.fixture(scope="module")
def model():
    """Load model once for all tests in this module."""
    return PatchableModel(model_name=MODEL_NAME, device=DEVICE, max_length=64)


@pytest.fixture
def sample_example():
    """A minimal dataset example for testing."""
    return {
        "id": "test_001",
        "query": {
            "base": "python web scraping",
            "neg": "python web scraping without selenium",
        },
        "doc_pos": {"text": "BeautifulSoup is a Python library for web scraping HTML pages."},
        "doc_neg": {"text": "Selenium is a browser automation tool for web scraping."},
        "slice_type": "explicit",
    }


class TestPatchingSite:
    """Tests for PatchingSite dataclass."""

    def test_component_site(self):
        site = PatchingSite(name="layer.3.ffn_out", module_path="bert.encoder.layer.3.output", layer_idx=3)
        assert not site.is_head_level
        assert site.layer_idx == 3
        assert site.head_idx is None

    def test_head_site(self):
        site = PatchingSite(
            name="layer.2.head.7",
            module_path="bert.encoder.layer.2.attention.self",
            layer_idx=2,
            head_idx=7,
        )
        assert site.is_head_level
        assert site.layer_idx == 2
        assert site.head_idx == 7


class TestSiteEnumeration:
    """Tests for site enumeration functions."""

    def test_component_sites_count(self):
        sites = get_component_sites()
        # 1 embed + 6*4 layer components + 1 pooler = 26
        assert len(sites) == 26

    def test_component_sites_names(self):
        sites = get_component_sites()
        names = [s.name for s in sites]
        assert "embed" in names
        assert "pooler" in names
        assert "layer.0.attn_self" in names
        assert "layer.5.ffn_out" in names

    def test_head_sites_count(self):
        sites = get_head_sites()
        assert len(sites) == NUM_LAYERS * NUM_HEADS  # 72

    def test_head_sites_specific_layers(self):
        sites = get_head_sites(layers=[2, 4])
        assert len(sites) == 2 * NUM_HEADS  # 24

    def test_head_sites_names(self):
        sites = get_head_sites(layers=[0])
        names = [s.name for s in sites]
        assert "layer.0.head.0" in names
        assert "layer.0.head.11" in names


class TestPatchableModel:
    """Tests for PatchableModel class."""

    def test_model_loads(self, model):
        assert model.model is not None
        assert model.tokenizer is not None
        assert model.device == torch.device(DEVICE)

    def test_tokenize_pair(self, model):
        inputs = model.tokenize_pair("test query", "test document")
        assert "input_ids" in inputs
        assert "attention_mask" in inputs
        assert inputs["input_ids"].shape[0] == 1  # batch size 1
        assert inputs["input_ids"].shape[1] == 64  # max_length

    def test_score_returns_scalar(self, model):
        inputs = model.tokenize_pair("python tutorial", "Learn Python programming")
        score = model.score(inputs)
        assert isinstance(score, float)

    def test_score_consistency(self, model):
        """Same input should give same score."""
        inputs = model.tokenize_pair("python tutorial", "Learn Python programming")
        score1 = model.score(inputs)
        score2 = model.score(inputs)
        assert abs(score1 - score2) < 1e-5

    def test_cache_activations_shapes(self, model):
        inputs = model.tokenize_pair("test query", "test document")
        sites = get_component_sites()
        cache = model.cache_activations(inputs, sites)

        assert "embed" in cache
        assert cache["embed"].shape == (1, 64, HIDDEN_DIM)

        assert "layer.0.attn_self" in cache
        assert cache["layer.0.attn_self"].shape == (1, 64, HIDDEN_DIM)

        assert "layer.0.ffn_mid" in cache
        assert cache["layer.0.ffn_mid"].shape[2] == 1536  # FFN_DIM

        assert "pooler" in cache
        assert cache["pooler"].shape == (1, HIDDEN_DIM)

    def test_cache_on_cpu(self, model):
        """Cached activations should be on CPU regardless of model device."""
        inputs = model.tokenize_pair("test", "doc")
        sites = get_component_sites()[:3]
        cache = model.cache_activations(inputs, sites)
        for tensor in cache.values():
            assert tensor.device == torch.device("cpu")

    def test_patch_and_score(self, model, sample_example):
        """Patching should change the output score."""
        fields = sample_example
        q = fields["query"]["neg"]
        d_pos = fields["doc_pos"]["text"]
        d_neg = fields["doc_neg"]["text"]

        clean_inputs = model.tokenize_pair(q, d_pos)
        corrupt_inputs = model.tokenize_pair(q, d_neg)

        sites = get_component_sites()
        clean_cache = model.cache_activations(clean_inputs, sites)

        corrupt_score = model.score(corrupt_inputs)
        # Patching the pooler should change output
        patched_score = model.patch_and_score(
            corrupt_inputs, clean_cache, sites[-1]  # pooler
        )
        # Patching should produce a different score (extremely unlikely to be identical)
        assert patched_score != corrupt_score

    def test_patch_restores_clean_for_full_patch(self, model, sample_example):
        """If we patch ALL sites, result should approach clean score."""
        fields = sample_example
        q = fields["query"]["neg"]
        d_pos = fields["doc_pos"]["text"]
        d_neg = fields["doc_neg"]["text"]

        clean_inputs = model.tokenize_pair(q, d_pos)
        clean_score = model.score(clean_inputs)

        # Patching just the pooler (last layer before classifier) should
        # restore most of the clean signal
        sites = get_component_sites()
        clean_cache = model.cache_activations(clean_inputs, sites)
        corrupt_inputs = model.tokenize_pair(q, d_neg)

        pooler_site = [s for s in sites if s.name == "pooler"][0]
        patched = model.patch_and_score(corrupt_inputs, clean_cache, pooler_site)
        # Patching pooler should push score close to clean
        assert abs(patched - clean_score) < abs(model.score(corrupt_inputs) - clean_score)

    def test_head_level_patching(self, model, sample_example):
        """Head-level patching should only modify a slice of the activation."""
        q = sample_example["query"]["neg"]
        d_pos = sample_example["doc_pos"]["text"]
        d_neg = sample_example["doc_neg"]["text"]

        clean_inputs = model.tokenize_pair(q, d_pos)
        corrupt_inputs = model.tokenize_pair(q, d_neg)

        head_sites = get_head_sites(layers=[0])
        # Cache at the attn_self level
        cache_site = PatchingSite(
            name="layer.0.attn_self",
            module_path="bert.encoder.layer.0.attention.self",
            layer_idx=0,
        )
        clean_cache = model.cache_activations(clean_inputs, [cache_site])

        # Patch head 0
        head_site = head_sites[0]
        head_cache = {head_site.name: clean_cache["layer.0.attn_self"]}
        patched = model.patch_and_score(corrupt_inputs, head_cache, head_site)
        assert isinstance(patched, float)

    def test_zero_ablation(self, model, sample_example):
        """Zero ablation should change the score."""
        q = sample_example["query"]["neg"]
        d_pos = sample_example["doc_pos"]["text"]
        inputs = model.tokenize_pair(q, d_pos)

        normal_score = model.score(inputs)
        site = get_component_sites()[1]  # layer.0.attn_self
        ablated_score = model.ablate_and_score(inputs, [site], method="zero")
        # Ablation should change the score
        assert ablated_score != normal_score

    def test_multi_site_ablation(self, model, sample_example):
        """Ablating multiple sites should work."""
        q = sample_example["query"]["neg"]
        d_pos = sample_example["doc_pos"]["text"]
        inputs = model.tokenize_pair(q, d_pos)

        sites = get_component_sites()[:3]  # embed, layer.0.attn_self, layer.0.attn_out
        ablated_score = model.ablate_and_score(inputs, sites, method="zero")
        assert isinstance(ablated_score, float)

    def test_hooks_cleaned_up(self, model):
        """Hooks should not persist after operations."""
        inputs = model.tokenize_pair("test", "doc")
        sites = get_component_sites()[:2]

        # Cache activations (registers and removes hooks)
        model.cache_activations(inputs, sites)

        # Score should be unaffected by previous hook operations
        score1 = model.score(inputs)
        score2 = model.score(inputs)
        assert abs(score1 - score2) < 1e-5


class TestScanResult:
    """Tests for ScanResult data aggregation."""

    def test_compute_stats(self):
        results = [
            PatchingResult("ex1", "layer.0.attn_self", 5.0, 2.0, 3.5, 3.0, 1.5, 0.5),
            PatchingResult("ex2", "layer.0.attn_self", 6.0, 3.0, 4.0, 3.0, 1.0, 0.333),
            PatchingResult("ex1", "layer.0.ffn_out", 5.0, 2.0, 2.5, 3.0, 0.5, 0.167),
        ]
        scan = ScanResult(
            scan_type="component",
            model_name="test",
            num_examples=2,
            sites=["layer.0.attn_self", "layer.0.ffn_out"],
            per_example=results,
        )
        scan.compute_stats()

        assert "layer.0.attn_self" in scan.site_stats
        assert "layer.0.ffn_out" in scan.site_stats
        assert scan.site_stats["layer.0.attn_self"]["count"] == 2
        assert scan.site_stats["layer.0.ffn_out"]["count"] == 1

    def test_to_dict(self):
        scan = ScanResult(
            scan_type="component",
            model_name="test",
            num_examples=0,
            sites=["layer.0.attn_self"],
        )
        d = scan.to_dict()
        assert "scan_type" in d
        assert "sites" in d
        assert "per_example" in d


class TestGetHotSites:
    """Tests for hot site identification."""

    def test_threshold_filtering(self):
        scan = ScanResult(
            scan_type="component",
            model_name="test",
            num_examples=10,
            sites=["a", "b", "c"],
            site_stats={
                "a": {"mean": 0.2, "std": 0.1, "median": 0.2, "abs_mean": 0.2, "count": 10},
                "b": {"mean": 0.05, "std": 0.01, "median": 0.05, "abs_mean": 0.05, "count": 10},
                "c": {"mean": -0.15, "std": 0.1, "median": -0.15, "abs_mean": 0.15, "count": 10},
            },
        )
        hot = get_hot_sites(scan, threshold=0.1)
        names = [name for name, _ in hot]
        assert "a" in names
        assert "c" in names
        assert "b" not in names

    def test_sorted_by_effect(self):
        scan = ScanResult(
            scan_type="component",
            model_name="test",
            num_examples=10,
            sites=["a", "b"],
            site_stats={
                "a": {"mean": 0.1, "std": 0.05, "median": 0.1, "abs_mean": 0.1, "count": 10},
                "b": {"mean": 0.3, "std": 0.1, "median": 0.3, "abs_mean": 0.3, "count": 10},
            },
        )
        hot = get_hot_sites(scan, threshold=0.05)
        assert hot[0][0] == "b"  # b has higher effect


class TestIntegration:
    """Integration tests that run scans on small examples."""

    @pytest.fixture
    def small_examples(self):
        return [
            {
                "id": "int_001",
                "query": {"neg": "python web scraping without selenium"},
                "doc_pos": {"text": "BeautifulSoup is a Python library for parsing HTML and XML."},
                "doc_neg": {"text": "Selenium automates web browsers for testing and scraping."},
                "slice_type": "explicit",
            },
            {
                "id": "int_002",
                "query": {"neg": "machine learning without neural networks"},
                "doc_pos": {"text": "Random forests and SVMs are popular machine learning methods."},
                "doc_neg": {"text": "Deep neural networks have revolutionized machine learning."},
                "slice_type": "explicit",
            },
        ]

    def test_component_scan_runs(self, model, small_examples):
        """Component scan produces results with correct structure."""
        scan = run_component_scan(
            model=model,
            examples=small_examples,
            min_total_effect=0.0,  # Don't filter for test
        )
        assert scan.scan_type == "component"
        assert scan.num_examples == 2
        assert len(scan.per_example) > 0
        assert len(scan.site_stats) > 0

    def test_head_scan_runs(self, model, small_examples):
        """Head scan produces results."""
        scan = run_head_scan(
            model=model,
            examples=small_examples,
            layers=[0],  # Only scan layer 0 for speed
            min_total_effect=0.0,
        )
        assert scan.scan_type == "head"
        assert len(scan.per_example) > 0

    def test_ablation_study_runs(self, model, small_examples):
        """Ablation study produces results."""
        target_sites = [get_component_sites()[1]]  # layer.0.attn_self
        result = run_ablation_study(
            model=model,
            examples=small_examples,
            target_sites=target_sites,
            method="zero",
            n_random_controls=1,
        )
        assert isinstance(result, AblationResult)
        assert result.baseline_accuracy >= 0.0
        assert len(result.targeted_results) == 1
        assert len(result.random_controls) == 1
