"""Tests for country-aware candidate blocking and recall evaluation."""

import pandas as pd

from src.blocking import evaluate_blocking_recall, generate_candidate_pairs


def _frame(rows):
	"""Build a compact preprocessed source frame for blocking tests."""
	return pd.DataFrame(
		rows,
		columns=[
			"entity_id",
			"normalized_name",
			"address_tokens",
			"is_non_latin_name",
			"normalized_country",
		],
	)


def test_candidate_generation_only_compares_source1_with_other_sources():
	"""Candidates are country-local and never contain a Source2/3 cross-pair."""
	source1 = _frame(
		[("S1-1", "acme widgets", ["42", "oak"], False, "us")]
	)
	source2 = _frame(
		[
			("S2-1", "acme widgets", ["42", "oak"], False, "us"),
			("S2-2", "acme widgets", ["42", "oak"], False, "france"),
		]
	)
	source3 = _frame(
		[("S3-1", "acme widgets", ["42", "oak"], False, "us")]
	)

	candidates = generate_candidate_pairs(source1, source2, source3)

	assert set(candidates["candidate_entity_id"]) == {"S2-1", "S3-1"}
	assert set(candidates["source1_entity_id"]) == {"S1-1"}
	assert set(candidates["candidate_source"]) == {"source2", "source3"}
	assert candidates["block_reason"].eq("both").all()
	assert generate_candidate_pairs(
		source1, source2, source3, min_shared_address_tokens=2
	)["block_reason"].eq("both").all()


def test_address_fallback_finds_non_latin_target_and_threshold_is_configurable():
	"""A non-Latin endpoint uses address tokens, including a two-token option."""
	source1 = _frame(
		[("S1-1", "market house", ["9", "mumbai", "garden"], False, "india")]
	)
	source2 = _frame([])
	source3 = _frame(
		[("S3-1", "भारत उद्योग", ["9", "mumbai", "garden"], True, "india")]
	)

	candidates = generate_candidate_pairs(source1, source2, source3)
	assert candidates.loc[0, "block_reason"] == "address_token"
	assert len(generate_candidate_pairs(source1, source2, source3, min_shared_address_tokens=2)) == 1
	assert len(generate_candidate_pairs(source1, source2, source3, min_shared_address_tokens=4)) == 0


def test_recall_reports_misses_and_non_latin_stratum():
	"""Recall includes missed matches and classifies either non-Latin endpoint."""
	source1 = _frame(
		[
			("S1-1", "plain company", ["1", "oak"], False, "us"),
			("S1-2", "short", ["2", "market"], False, "us"),
		]
	)
	source2 = _frame([("S2-1", "plain company", ["1", "oak"], False, "us")])
	source3 = _frame([("S3-1", "भारत उद्योग", ["2", "market"], True, "us")])
	candidates = generate_candidate_pairs(source1, source2, source3)
	ground_truth = pd.DataFrame(
		{
			"source1_entity_id": ["S1-1", "S1-2"],
			"matched_entity_ids": ["S2-1,S3-missing", "S3-1"],
		}
	)

	result = evaluate_blocking_recall(
		candidates,
		ground_truth,
		source1_df=source1,
		source2_df=source2,
		source3_df=source3,
	)

	assert result["true_match_count"] == 3
	assert result["recovered_true_match_count"] == 2
	assert result["overall_recall_percent"] == 100 * 2 / 3
	assert result["recall_by_non_latin_name"]["involved"]["true_matches"] == 1
	assert result["source1_entities_with_missed_matches"] == 1
	assert result["missed_examples"][0]["normalized_name"] == "plain company"