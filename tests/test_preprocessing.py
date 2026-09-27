"""Tests for entity-resolution preprocessing and normalization."""

import pandas as pd

from src.preprocessing import (
    extract_address_tokens,
    is_non_latin_script,
    normalize_address,
    normalize_business_name,
    normalize_country,
    normalize_unicode,
    preprocess_dataframe,
)


def test_normalize_unicode_removes_latin_diacritics():
    """Latin diacritics should be removed without changing the base letters."""
    assert normalize_unicode("Córp") == "Corp"


def test_normalize_unicode_preserves_non_latin_script():
    """NFKD-normalized Devanagari should not be transliterated or discarded."""
    text = "भारत"
    assert normalize_unicode(text) == text


def test_is_non_latin_script_detects_devanagari():
    """Script detection distinguishes Devanagari from ordinary English."""
    assert is_non_latin_script("भारत") is True
    assert is_non_latin_script("Acme Private Limited") is False


def test_normalize_business_name_canonicalizes_private_limited_suffix():
    """Pvt. Ltd. and Private Limited should share one retained suffix token."""
    assert normalize_business_name("Acme Pvt. Ltd.") == normalize_business_name(
        "Acme Private Limited"
    ) == "acme private_limited"


def test_normalize_business_name_canonicalizes_private_limited_variants():
    """Common Pvt/Private and Ltd/Limited combinations share one suffix token."""
    variants = [
        "Shree Infracon PRIVATE LIMITED",
        "Shree Infracon, Pvt. Ltd.",
        "Shree Infracon Private Ltd",
        "Shree Infracon PVT Limited",
    ]

    assert {normalize_business_name(name) for name in variants} == {
        "shree infracon private_limited"
    }


def test_normalize_business_name_preserves_non_latin_marks():
    """Name cleanup retains dependent vowel marks in non-Latin scripts."""
    assert normalize_business_name("राम कंपनी") == "राम कंपनी"


def test_normalize_address_handles_none():
    """Missing addresses normalize to an empty string."""
    assert normalize_address(None) == ""


def test_extract_address_tokens_includes_house_number():
    """Address token extraction retains salient numeric identifiers."""
    assert extract_address_tokens("Plot 42, MG Road, Bengaluru") == [
        "plot",
        "42",
        "road",
        "bengaluru",
    ]


def test_extract_address_tokens_preserves_non_latin_marks():
    """Unicode word tokens keep their combining marks intact."""
    assert extract_address_tokens("रामनगर मार्ग 42") == ["रामनगर", "मार्ग", "42"]


def test_normalize_country_accepts_unseen_country():
    """Country normalization does not validate against a training-time list."""
    assert normalize_country(" France ") == "france"


def test_preprocess_dataframe_adds_columns_and_handles_missing_values():
    """Preprocessing supports missing addresses and non-Latin names."""
    source = pd.DataFrame(
        {
            "entity_id": ["1", "2", "3"],
            "business_name": ["Córp Pvt. Ltd.", "भारत कंपनी", "Short"],
            "business_address": ["12 Main St.", float("nan"), ""],
            "country": ["US", "India", "France"],
        }
    )

    result = preprocess_dataframe(source)

    expected_columns = {
        "normalized_name",
        "normalized_address",
        "address_tokens",
        "is_non_latin_name",
        "normalized_country",
    }
    assert expected_columns.issubset(result.columns)
    assert result.loc[0, "normalized_name"] == "corporation private_limited"
    assert result.loc[0, "normalized_address"] == "12 main street"
    assert result.loc[0, "address_tokens"] == ["12", "main", "street"]
    assert result.loc[1, "normalized_address"] == ""
    assert result.loc[1, "address_tokens"] == []
    assert bool(result.loc[1, "is_non_latin_name"])
    assert result.loc[2, "normalized_country"] == "france"
    assert source.loc[0, "business_name"] == "Córp Pvt. Ltd."


def test_preprocess_dataframe_handles_empty_frame():
    """An empty input still receives the derived columns."""
    result = preprocess_dataframe(pd.DataFrame())
    assert len(result) == 0
    assert "normalized_name" in result.columns
    assert "address_tokens" in result.columns