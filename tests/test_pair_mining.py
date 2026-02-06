"""
Tests for pair mining functionality.

Run with: pytest tests/test_pair_mining.py
"""

import pytest

from constraintsuite.pair_mining import (
    contains_y,
    y_is_negated_nearby,
    mine_pair,
    mine_minpair,
    classify_pair_slice,
    find_y_positions,
)


class TestContainsY:
    """Tests for contains_y function."""

    def test_contains_y_basic(self):
        """Test basic Y detection."""
        text = "This document mentions selenium for web automation"
        assert contains_y(text, ["selenium"]) is True

    def test_contains_y_not_present(self):
        """Test when Y is not present."""
        text = "This document is about Python and BeautifulSoup"
        assert contains_y(text, ["selenium"]) is False

    def test_contains_y_case_insensitive(self):
        """Test case-insensitive matching."""
        text = "SELENIUM is great for testing"
        assert contains_y(text, ["selenium"]) is True

        text = "selenium is lowercase"
        assert contains_y(text, ["SELENIUM"]) is True

    def test_contains_y_multiple_forms(self):
        """Test matching multiple surface forms."""
        text = "This talks about Se which is a browser driver"
        y_forms = ["selenium", "Se", "webdriver"]
        assert contains_y(text, y_forms) is True

    def test_contains_y_no_partial_match(self):
        """Test that word boundary matching prevents partial matches."""
        text = "The seleniumwebdriver is popular"
        assert contains_y(text, ["selenium"]) is False

    def test_contains_y_empty_text(self):
        """Test empty text."""
        assert contains_y("", ["selenium"]) is False

    def test_contains_y_empty_forms(self):
        """Test empty forms list."""
        assert contains_y("some text", []) is False


class TestFindYPositions:
    """Tests for find_y_positions function."""

    def test_find_single_position(self):
        """Test finding single occurrence."""
        text = "I use selenium for testing"
        positions = find_y_positions(text, "selenium")
        assert len(positions) == 1
        assert positions[0] == (6, 14)  # (start, end) tuple

    def test_find_multiple_positions(self):
        """Test finding multiple occurrences."""
        text = "selenium is great, selenium is fast"
        positions = find_y_positions(text, "selenium")
        assert len(positions) == 2
        assert positions[0] == (0, 8)
        assert positions[1] == (19, 27)

    def test_find_case_insensitive(self):
        """Test case insensitive finding."""
        text = "SELENIUM and selenium and Selenium"
        positions = find_y_positions(text, "selenium")
        assert len(positions) == 3


class TestYIsNegatedNearby:
    """Tests for y_is_negated_nearby function."""

    def test_explicit_no(self):
        """Test 'no Y' pattern."""
        text = "This recipe contains no peanuts for safety"
        assert y_is_negated_nearby(text, "peanuts") is True

    def test_without_pattern(self):
        """Test 'without Y' pattern."""
        text = "Python web scraping without selenium is possible"
        assert y_is_negated_nearby(text, "selenium") is True

    def test_free_suffix(self):
        """Test 'Y-free' pattern."""
        text = "This is a gluten-free recipe for bread"
        assert y_is_negated_nearby(text, "gluten") is True

    def test_free_of_pattern(self):
        """Test 'free of Y' pattern."""
        text = "This recipe is free of peanuts and safe"
        assert y_is_negated_nearby(text, "peanuts") is True

    def test_not_pattern(self):
        """Test 'not Y' pattern."""
        text = "This does not contain selenium"
        assert y_is_negated_nearby(text, "selenium") is True

    def test_exclude_pattern(self):
        """Test 'excluding Y' pattern."""
        text = "All browsers excluding selenium are supported"
        assert y_is_negated_nearby(text, "selenium") is True

    def test_no_negation(self):
        """Test when Y is not negated."""
        text = "Thai peanut noodles recipe with peanut sauce"
        assert y_is_negated_nearby(text, "peanut") is False

    def test_no_negation_affirmative(self):
        """Test affirmative mention."""
        text = "Selenium is the best tool for browser automation"
        assert y_is_negated_nearby(text, "selenium") is False

    def test_negation_far_away(self):
        """Test negation too far from Y (beyond window)."""
        # Default window is 40 chars
        text = "no " + "x" * 50 + " selenium"
        assert y_is_negated_nearby(text, "selenium", window=40) is False

    def test_negation_within_window(self):
        """Test negation within window."""
        text = "no " + "x" * 30 + " selenium"
        assert y_is_negated_nearby(text, "selenium", window=40) is True

    def test_lacks_pattern(self):
        """Test 'lacks Y' pattern."""
        text = "This browser lacks selenium support"
        assert y_is_negated_nearby(text, "selenium") is True

    def test_absent_pattern(self):
        """Test 'absent Y' pattern."""
        text = "With selenium absent, we use other tools"
        assert y_is_negated_nearby(text, "selenium") is True


class TestMinePair:
    """Tests for mine_pair function."""

    def test_finds_valid_pair(self):
        """Test that valid pairs are found."""
        candidates = [
            {"id": "doc1", "text": "Selenium is great for browser automation testing"},
            {"id": "doc2", "text": "BeautifulSoup is great for web scraping without selenium"},
            {"id": "doc3", "text": "Python requests library for HTTP calls"},
        ]
        y_forms = ["selenium"]

        pair = mine_pair(candidates, y_forms)

        assert pair is not None
        assert pair.doc_neg["id"] == "doc1"  # Contains selenium affirmatively
        assert pair.doc_pos["id"] in ["doc2", "doc3"]  # Either negates or omits

    def test_returns_none_no_violators(self):
        """Test returns None when no violators found."""
        candidates = [
            {"id": "doc1", "text": "BeautifulSoup for scraping"},
            {"id": "doc2", "text": "Requests library for HTTP"},
            {"id": "doc3", "text": "No selenium needed here"},  # Negated, not a violator
        ]
        y_forms = ["selenium"]

        pair = mine_pair(candidates, y_forms)
        assert pair is None

    def test_returns_none_no_satisfiers(self):
        """Test returns None when no satisfiers found."""
        candidates = [
            {"id": "doc1", "text": "Selenium is great"},
            {"id": "doc2", "text": "Use selenium for testing"},
            {"id": "doc3", "text": "Selenium webdriver rocks"},
        ]
        y_forms = ["selenium"]

        pair = mine_pair(candidates, y_forms)
        assert pair is None

    def test_prefers_similar_rank(self):
        """Test that pairs with similar BM25 rank are preferred."""
        # Candidates are in BM25 rank order (index = rank)
        candidates = [
            {"id": "doc0", "text": "Selenium browser automation"},  # rank 0, violator
            {"id": "doc1", "text": "Python web scraping basics"},   # rank 1, satisfier
            {"id": "doc2", "text": "HTTP requests in Python"},      # rank 2, satisfier
            {"id": "doc3", "text": "BeautifulSoup tutorial"},       # rank 3, satisfier
        ]
        y_forms = ["selenium"]

        pair = mine_pair(candidates, y_forms)

        assert pair is not None
        assert pair.doc_neg["id"] == "doc0"
        # Should prefer doc1 (closest rank to doc0)
        assert pair.doc_pos["id"] == "doc1"

    def test_prefers_explicit_negation(self):
        """Test preferring doc_pos with explicit negation."""
        candidates = [
            {"id": "doc0", "text": "Selenium is the best tool"},
            {"id": "doc1", "text": "Python without any mention"},  # omission
            {"id": "doc2", "text": "Scraping without selenium"},   # explicit negation
        ]
        y_forms = ["selenium"]

        pair = mine_pair(candidates, y_forms, prefer_explicit=True)

        assert pair is not None
        assert pair.doc_pos["id"] == "doc2"
        assert pair.slice_type == "explicit"


class TestMineMinpair:
    """Tests for mine_minpair function."""

    def test_insert_no_edit(self):
        """Test 'insert_no' edit type."""
        doc = {"doc_id": "doc1", "text": "Use selenium for browser testing."}

        minpair = mine_minpair(doc, "selenium", edit_type="insert_no")

        assert minpair is not None
        assert "no selenium" in minpair.doc_pos["text"].lower()
        assert minpair.doc_neg["text"] == doc["text"]
        assert minpair.slice_type == "minpairs"

    def test_add_free_edit(self):
        """Test 'add_free' edit type."""
        doc = {"doc_id": "doc1", "text": "Use selenium for browser testing."}

        minpair = mine_minpair(doc, "selenium", edit_type="add_free")

        assert minpair is not None
        assert "selenium-free" in minpair.doc_pos["text"].lower()

    def test_minpair_preserves_id(self):
        """Test that minpair preserves original doc ID with modification."""
        doc = {"doc_id": "original_id", "text": "Selenium is great."}

        minpair = mine_minpair(doc, "selenium", edit_type="insert_no")

        assert minpair is not None
        assert "original_id" in minpair.doc_pos["doc_id"]
        assert minpair.doc_neg["doc_id"] == "original_id"


class TestClassifyPairSlice:
    """Tests for classify_pair_slice function."""

    def test_minpairs_classification(self):
        """Test MinPairs slice classification (high similarity triggers minpairs)."""
        doc_pos = {"text": "Use no selenium for testing"}
        doc_neg = {"text": "Use selenium for testing"}

        slice_type = classify_pair_slice(doc_pos, doc_neg, ["selenium"])
        assert slice_type == "minpairs"

    def test_minpairs_via_is_edited(self):
        """Test MinPairs via is_edited flag."""
        doc_pos = {"text": "Use no selenium for testing", "is_edited": True}
        doc_neg = {"text": "Selenium is great for web scraping automation"}

        slice_type = classify_pair_slice(doc_pos, doc_neg, ["selenium"])
        assert slice_type == "minpairs"

    def test_explicit_classification(self):
        """Test ExplicitMention slice classification."""
        doc_pos = {"text": "Web scraping without selenium using BeautifulSoup"}
        doc_neg = {"text": "Selenium is great for web scraping automation"}

        slice_type = classify_pair_slice(doc_pos, doc_neg, ["selenium"])
        assert slice_type == "explicit"

    def test_omission_classification(self):
        """Test Omission slice classification."""
        doc_pos = {"text": "BeautifulSoup is great for web scraping"}
        doc_neg = {"text": "Selenium is great for web scraping automation"}

        slice_type = classify_pair_slice(doc_pos, doc_neg, ["selenium"])
        assert slice_type == "omission"


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_candidates(self):
        """Test with empty candidate list."""
        pair = mine_pair([], ["selenium"])
        assert pair is None

    def test_unicode_text(self):
        """Test with unicode characters."""
        text = "セレニウム selenium тест"
        assert contains_y(text, ["selenium"]) is True

    def test_special_characters_in_y(self):
        """Test Y with special regex characters."""
        text = "Use C++ for programming"
        # Should handle the + characters properly
        assert contains_y(text, ["C++"]) is True

    def test_multiword_y(self):
        """Test multi-word Y terms."""
        text = "Machine learning is great"
        assert contains_y(text, ["machine learning"]) is True

    def test_hyphenated_y(self):
        """Test hyphenated Y terms."""
        text = "This is a gluten-free recipe"
        assert contains_y(text, ["gluten-free"]) is True
        # Also test the component
        assert contains_y(text, ["gluten"]) is True
