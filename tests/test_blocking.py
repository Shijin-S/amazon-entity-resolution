"""Tests for country-aware candidate blocking and recall evaluation."""

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pyarrow.parquet as pq
import pytest

from src import blocking
from src.blocking import (
	MemorySafetyLimitExceeded,
	evaluate_blocking_recall,
	generate_candidate_pairs,
)


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


def _write_sources(tmp_path, source1, source2, source3):
	paths = []
	for source_name, frame in (("source1", source1), ("source2", source2), ("source3", source3)):
		path = tmp_path / f"{source_name}.parquet"
		frame.to_parquet(path, index=False)
		paths.append(str(path))
	return paths


def test_candidate_generation_only_compares_source1_with_other_sources(tmp_path):
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

	paths = _write_sources(tmp_path, source1, source2, source3)
	candidates = generate_candidate_pairs(*paths, return_candidate_dataframe=True)

	assert set(candidates["candidate_entity_id"]) == {"S2-1", "S3-1"}
	assert set(candidates["source1_entity_id"]) == {"S1-1"}
	assert set(candidates["candidate_source"]) == {"source2", "source3"}
	assert candidates["block_reason"].eq("both").all()
	assert generate_candidate_pairs(
		*paths, min_shared_address_tokens=2, return_candidate_dataframe=True
	)["block_reason"].eq("both").all()


def test_candidate_generation_can_restrict_countries(tmp_path):
	source1 = _frame(
		[
			("S1-us", "acme widgets", ["42", "oak"], False, "us"),
			("S1-fr", "acme widgets", ["42", "oak"], False, "france"),
		]
	)
	source2 = _frame(
		[
			("S2-us", "acme widgets", ["42", "oak"], False, "us"),
			("S2-fr", "acme widgets", ["42", "oak"], False, "france"),
		]
	)
	source3 = _frame(
		[
			("S3-us", "acme widgets", ["42", "oak"], False, "us"),
			("S3-fr", "acme widgets", ["42", "oak"], False, "france"),
		]
	)
	paths = _write_sources(tmp_path, source1, source2, source3)

	candidates = generate_candidate_pairs(
		*paths, countries=["us"], return_candidate_dataframe=True
	)

	assert set(candidates["source1_entity_id"]) == {"S1-us"}
	assert set(candidates["candidate_entity_id"]) == {"S2-us", "S3-us"}


def test_candidate_generation_matches_across_multiple_batches(tmp_path):
	source1 = _frame(
		[(f"S1-{row}", "acme widgets", ["42", "oak"], False, "us") for row in range(5)]
	)
	source2 = _frame(
		[(f"S2-{row}", "acme widgets", ["42", "oak"], False, "us") for row in range(6)]
	)
	source3 = _frame(
		[(f"S3-{row}", "acme widgets", ["42", "oak"], False, "us") for row in range(5)]
	)
	paths = _write_sources(tmp_path, source1, source2, source3)

	batched = generate_candidate_pairs(*paths, batch_size=2, return_candidate_dataframe=True)
	single_batch = generate_candidate_pairs(*paths, batch_size=100, return_candidate_dataframe=True)
	streamed_path = generate_candidate_pairs(
		*paths,
		batch_size=2,
		match_batch_size=2,
		candidate_output_dir=str(tmp_path / "streamed"),
		return_candidate_dataframe=False,
	)
	streamed = pd.read_parquet(streamed_path)
	assert pq.ParquetFile(Path(streamed_path) / "candidates.parquet").metadata.num_row_groups > 1

	columns = ["source1_entity_id", "candidate_entity_id", "candidate_source", "block_reason"]
	batched_rows = batched.sort_values(columns).reset_index(drop=True)
	single_batch_rows = single_batch.sort_values(columns).reset_index(drop=True)
	pd.testing.assert_frame_equal(batched_rows, single_batch_rows)
	streamed_rows = streamed.sort_values(columns).reset_index(drop=True)
	pd.testing.assert_frame_equal(batched_rows, streamed_rows)
	assert len(batched) == 5 * (6 + 5)
	assert batched["block_reason"].eq("both").all()


def test_high_frequency_address_token_is_dropped(tmp_path, caplog):
	"""An address token over the posting cap is dropped rather than expanded."""
	caplog.set_level("INFO")
	source1 = _frame([("S1-1", "x", ["oak"], False, "us")])
	source2 = _frame(
		[(f"S2-{row}", "x", ["oak", f"place{row}"], False, "us") for row in range(3)]
	)
	source3 = _frame([])
	paths = _write_sources(tmp_path, source1, source2, source3)

	candidates = generate_candidate_pairs(
		*paths,
		max_postings_per_token=2,
		batch_size=2,
		return_candidate_dataframe=True,
	)

	assert candidates.empty
	assert "Dropped address token 'oak': original posting-list length=3" in caplog.text
	assert "top_address_tokens" in caplog.text

	name_source1 = _frame([("S1-name", "acme widgets", ["unique1"], False, "us")])
	name_source2 = _frame(
		[
			(f"S2-name-{row}", "acme widgets", [f"unique{row + 2}"], False, "us")
			for row in range(3)
		]
	)
	name_cap_directory = tmp_path / "name-cap"
	name_cap_directory.mkdir()
	name_paths = _write_sources(name_cap_directory, name_source1, name_source2, source3)
	name_candidates = generate_candidate_pairs(
		*name_paths,
		max_postings_per_token=2,
		batch_size=2,
		return_candidate_dataframe=True,
	)
	assert name_candidates.empty
	assert "Dropped name block key ('acme', 'widgets'): original posting-list length=3" in caplog.text

	fallback_source1 = _frame([("S1-fallback", "acme widgets", ["oak"], False, "us")])
	fallback_source2 = _frame(
		[
			("S2-fallback-0", "acme widgets", ["oak"], False, "us"),
			("S2-fallback-1", "acme widgets", ["oak"], False, "us"),
			("S2-fallback-2", "acme widgets", ["pine"], False, "us"),
		]
	)
	fallback_directory = tmp_path / "name-cap-fallback"
	fallback_directory.mkdir()
	fallback_paths = _write_sources(
		fallback_directory, fallback_source1, fallback_source2, source3
	)
	fallback_candidates = generate_candidate_pairs(
		*fallback_paths,
		max_postings_per_token=2,
		batch_size=2,
		return_candidate_dataframe=True,
	)
	assert set(fallback_candidates["candidate_entity_id"]) == {
		"S2-fallback-0",
		"S2-fallback-1",
	}
	assert fallback_candidates["block_reason"].eq("address_token").all()


def test_memory_guard_stops_during_target_index_and_keeps_partial_directory(tmp_path):
	"""Index-phase RSS violations raise clearly and preserve the output location."""
	source1 = _frame(
		[
			("S1-1", "acme widgets", ["oak"], False, "us"),
			("S1-2", "acme widgets", ["pine"], False, "us"),
		]
	)
	source2 = _frame([("S2-1", "acme widgets", ["oak"], False, "us")])
	source3 = _frame([])
	paths = _write_sources(tmp_path, source1, source2, source3)

	with pytest.raises(MemorySafetyLimitExceeded, match="phase=target_index batch=1") as error:
		generate_candidate_pairs(
			*paths,
			candidate_output_dir=str(tmp_path / "partial"),
			return_candidate_dataframe=False,
			memory_safety_limit_mb=1,
		)

	assert Path(error.value.candidate_output_path).is_dir()
	assert "saved_candidate_rows=0" in str(error.value)


def test_memory_guard_stops_at_matching_batch(tmp_path, monkeypatch):
	"""Matching batches are guarded after target indexing has completed."""
	source1 = _frame(
		[
			("S1-1", "acme widgets", ["oak"], False, "us"),
			("S1-2", "acme widgets", ["pine"], False, "us"),
		]
	)
	source2 = _frame([("S2-1", "acme widgets", ["oak"], False, "us")])
	source3 = _frame([])
	paths = _write_sources(tmp_path, source1, source2, source3)
	rss_values = iter([100, 100, 100, 100, 100, 6_000])

	class FakeProcess:
		def memory_info(self):
			return SimpleNamespace(rss=next(rss_values) * 1024 * 1024)

	monkeypatch.setattr(blocking.psutil, "Process", FakeProcess)
	with pytest.raises(MemorySafetyLimitExceeded, match="phase=source1_match batch=2") as error:
		generate_candidate_pairs(
			*paths,
			candidate_output_dir=str(tmp_path / "matching-partial"),
			return_candidate_dataframe=False,
			memory_safety_limit_mb=5_000,
			match_batch_size=1,
		)
	partial_candidates = pd.read_parquet(error.value.candidate_output_path)
	assert partial_candidates["source1_entity_id"].tolist() == ["S1-1"]


def test_address_fallback_finds_non_latin_target_and_threshold_is_configurable(tmp_path):
	"""A non-Latin endpoint uses address tokens, including a two-token option."""
	source1 = _frame(
		[("S1-1", "market house", ["9", "mumbai", "garden"], False, "india")]
	)
	source2 = _frame([])
	source3 = _frame(
		[("S3-1", "भारत उद्योग", ["9", "mumbai", "garden"], True, "india")]
	)

	paths = _write_sources(tmp_path, source1, source2, source3)
	candidates = generate_candidate_pairs(*paths, return_candidate_dataframe=True)
	assert candidates.loc[0, "block_reason"] == "address_token"
	assert len(
		generate_candidate_pairs(
			*paths, min_shared_address_tokens=2, return_candidate_dataframe=True
		)
	) == 1
	assert len(
		generate_candidate_pairs(
			*paths, min_shared_address_tokens=4, return_candidate_dataframe=True
		)
	) == 0


def test_recall_reports_misses_and_non_latin_stratum(tmp_path):
	"""Recall includes missed matches and classifies either non-Latin endpoint."""
	source1 = _frame(
		[
			("S1-1", "plain company", ["1", "oak"], False, "us"),
			("S1-2", "short", ["2", "market"], False, "us"),
		]
	)
	source2 = _frame([("S2-1", "plain company", ["1", "oak"], False, "us")])
	source3 = _frame([("S3-1", "भारत उद्योग", ["2", "market"], True, "us")])
	paths = _write_sources(tmp_path, source1, source2, source3)
	candidates = generate_candidate_pairs(*paths, return_candidate_dataframe=True)
	ground_truth = pd.DataFrame(
		{
			"source1_entity_id": ["S1-1", "S1-2"],
			"matched_entity_ids": ["S2-1,S3-missing", "S3-1"],
		}
	)

	result = evaluate_blocking_recall(
		candidates,
		ground_truth,
		source1_path=paths[0],
		source2_path=paths[1],
		source3_path=paths[2],
	)

	assert result["true_match_count"] == 3
	assert result["recovered_true_match_count"] == 2
	assert result["overall_recall_percent"] == 100 * 2 / 3
	assert result["recall_by_non_latin_name"]["involved"]["true_matches"] == 1
	assert result["source1_entities_with_missed_matches"] == 1
	assert result["missed_examples"][0]["normalized_name"] == "plain company"