"""Country-aware candidate generation and blocking recall validation."""

import logging
from array import array
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)

_LEGAL_SUFFIX_TOKENS = frozenset(
	{
		"private_limited",
		"limited",
		"corporation",
		"llc",
		"inc",
		"incorporated",
		"plc",
		"lp",
		"llp",
		"company",
		"co",
		"gmbh",
		"sarl",
		"sa",
	}
)

_GENERIC_NAME_TOKENS = frozenset(
	{
		"agency",
		"associates",
		"business",
		"company",
		"construction",
		"enterprise",
		"enterprises",
		"global",
		"group",
		"industry",
		"industries",
		"international",
		"management",
		"services",
		"solution",
		"solutions",
		"technology",
		"technologies",
		"trading",
		"ventures",
	}
)

# Exclude the expanded address abbreviations as well as common address
# scaffolding; retaining those postings would create broad, low-value blocks.
_ADDRESS_STOPWORDS = frozenset(
	{
		"street",
		"road",
		"avenue",
		"boulevard",
		"lane",
		"drive",
		"highway",
		"parkway",
		"apartment",
		"suite",
		"floor",
		"plot",
		"house",
		"building",
		"number",
		"near",
		"opposite",
		"behind",
		"sector",
		"block",
		"district",
		"india",
	}
)

_OUTPUT_COLUMNS = [
	"source1_entity_id",
	"candidate_entity_id",
	"candidate_source",
	"block_reason",
]


def _name_block_key(name: object, minimum_length: int) -> tuple[str, str] | None:
	"""Return a cheap sorted endpoint signature, or None for weak names."""
	if name is None or pd.isna(name):
		return None
	text = str(name).strip().lower()
	if not text:
		return None

	# A short/generic name uses the broader address fallback. This avoids
	# widening a name block around weak tokens, at the cost of some extra address
	# comparisons for those records.
	tokens = text.split()
	meaningful_tokens = [
		token
		for token in tokens
		if token not in _LEGAL_SUFFIX_TOKENS and token not in _GENERIC_NAME_TOKENS
	]
	if sum(len(token) for token in meaningful_tokens) < minimum_length:
		return None
	if not meaningful_tokens:
		return None

	# Sorted first/last tokens are compact and tolerate word-order variation;
	# spelling variation in either endpoint can still miss, so uncertain names
	# are routed through address tokens instead.
	return tuple(sorted((meaningful_tokens[0], meaningful_tokens[-1])))


def _needs_address_fallback(name: object, is_non_latin: object, minimum_length: int) -> bool:
	"""Identify non-Latin, short, or low-information names for fallback."""
	return bool(is_non_latin) or _name_block_key(name, minimum_length) is None


def _filtered_address_tokens(value: object) -> set[str]:
	"""Return unique significant address tokens, excluding structural words."""
	if not isinstance(value, (list, tuple, set, frozenset, np.ndarray)):
		return set()
	return {
		str(token).strip().lower()
		for token in value
		if token is not None and str(token).strip().lower() not in _ADDRESS_STOPWORDS
	}


def _country_positions(frame: pd.DataFrame) -> dict[str, np.ndarray]:
	"""Map open-string country labels to positional row indexes."""
	countries = frame["normalized_country"].fillna("").to_numpy(copy=False)
	positions = pd.Series(np.arange(len(frame), dtype=np.int64))
	return positions.groupby(countries, sort=False).indices


def _record_arrays(frame: pd.DataFrame) -> dict[str, np.ndarray]:
	"""Expose only the columns needed by blocking as NumPy arrays."""
	return {
		"entity_id": frame["entity_id"].to_numpy(copy=False),
		"normalized_name": frame["normalized_name"].to_numpy(copy=False),
		"address_tokens": frame["address_tokens"].to_numpy(copy=False),
		"is_non_latin_name": frame["is_non_latin_name"].to_numpy(copy=False),
	}


def _unique_pair_codes(pair_codes: array, minimum_shared_tokens: int = 1) -> np.ndarray:
	"""Deduplicate encoded pairs and enforce the shared-token threshold."""
	if not pair_codes:
		return np.empty(0, dtype=np.uint64)
	values = np.frombuffer(pair_codes, dtype=np.uint64)
	if minimum_shared_tokens == 1:
		return np.unique(values)
	unique_codes, shared_counts = np.unique(values, return_counts=True)
	return unique_codes[shared_counts >= minimum_shared_tokens]


def _block_country_partition(
	source1_positions: np.ndarray,
	target_positions: np.ndarray,
	source1_records: dict[str, np.ndarray],
	target_records: dict[str, np.ndarray],
	candidate_source: str,
	name_min_length: int,
	min_shared_address_tokens: int,
) -> tuple[pd.DataFrame, int, int]:
	"""Generate and locally deduplicate pairs for one country/source partition."""
	target_count = len(target_positions)
	# A country has far fewer than 2^32 rows at the target dataset scale, so
	# compact posting arrays substantially reduce index memory vs Python integers.
	name_index: dict[tuple[str, str], array] = {}
	for target_local, target_global in enumerate(target_positions):
		if target_records["is_non_latin_name"][target_global]:
			continue
		key = _name_block_key(target_records["normalized_name"][target_global], name_min_length)
		if key is not None:
			postings = name_index.get(key)
			if postings is None:
				postings = array("I")
				name_index[key] = postings
			postings.append(target_local)

	name_pair_codes = array("Q")
	address_pair_codes = array("Q")
	for source1_local, source1_global in enumerate(source1_positions):
		if source1_records["is_non_latin_name"][source1_global]:
			continue
		key = _name_block_key(source1_records["normalized_name"][source1_global], name_min_length)
		if key is None:
			continue
		source1_address_tokens = _filtered_address_tokens(
			source1_records["address_tokens"][source1_global]
		)
		for target_local in name_index.get(key, ()):
			pair_code = source1_local * target_count + target_local
			name_pair_codes.append(pair_code)
			# Check address corroboration only among existing name candidates; a
			# global address join here would sharply increase recall but can explode
			# common-token blocks. The independent fallback below still covers
			# uncertain-name pairs.
			if min_shared_address_tokens == 1 or source1_address_tokens:
				target_address_tokens = _filtered_address_tokens(
					target_records["address_tokens"][target_positions[target_local]]
				)
				shared_count = len(source1_address_tokens & target_address_tokens)
				if shared_count >= min_shared_address_tokens:
					address_pair_codes.extend([pair_code] * shared_count)

	name_codes = _unique_pair_codes(name_pair_codes)
	del name_index, name_pair_codes

	# One token index stores all target postings plus a fallback-only posting
	# list. This finds pairs when either endpoint needs address matching without
	# keeping two complete indexes resident at once.
	address_index: dict[str, list[array]] = {}
	for target_local, target_global in enumerate(target_positions):
		fallback = _needs_address_fallback(
			target_records["normalized_name"][target_global],
			target_records["is_non_latin_name"][target_global],
			name_min_length,
		)
		for token in _filtered_address_tokens(target_records["address_tokens"][target_global]):
			postings = address_index.get(token)
			if postings is None:
				postings = [array("I"), array("I")]
				address_index[token] = postings
			postings[0].append(target_local)
			if fallback:
				postings[1].append(target_local)

	for source1_local, source1_global in enumerate(source1_positions):
		fallback = _needs_address_fallback(
			source1_records["normalized_name"][source1_global],
			source1_records["is_non_latin_name"][source1_global],
			name_min_length,
		)
		posting_position = 0 if fallback else 1
		for token in _filtered_address_tokens(source1_records["address_tokens"][source1_global]):
			postings = address_index.get(token)
			if postings is None:
				continue
			# Fallback Source1 records compare to all targets; ordinary Source1
			# records compare only to fallback targets. This captures mixed-script
			# true pairs without paying for address joins across every ordinary
			# record on both sides.
			for target_local in postings[posting_position]:
				address_pair_codes.append(source1_local * target_count + target_local)

	address_codes = _unique_pair_codes(address_pair_codes, min_shared_address_tokens)
	del address_index, address_pair_codes

	if not len(name_codes) and not len(address_codes):
		return pd.DataFrame(columns=_OUTPUT_COLUMNS), 0, 0

	all_codes = np.union1d(name_codes, address_codes)
	name_hits = np.isin(all_codes, name_codes, assume_unique=True)
	address_hits = np.isin(all_codes, address_codes, assume_unique=True)
	reasons = np.where(
		name_hits & address_hits,
		"both",
		np.where(name_hits, "name", "address_token"),
	)
	source1_local_positions = (all_codes // target_count).astype(np.int64)
	target_local_positions = (all_codes % target_count).astype(np.int64)
	source1_global_positions = source1_positions[source1_local_positions]
	target_global_positions = target_positions[target_local_positions]

	pairs = pd.DataFrame(
		{
			"source1_entity_id": source1_records["entity_id"][source1_global_positions],
			"candidate_entity_id": target_records["entity_id"][target_global_positions],
			"candidate_source": candidate_source,
			"block_reason": reasons,
		}
	)
	return pairs, len(name_codes), len(address_codes)


def _combine_block_reasons(reasons: Iterable[str]) -> str:
	"""Combine duplicate pair reasons into one stable strategy label."""
	has_name = any(reason in {"name", "both"} for reason in reasons)
	has_address = any(reason in {"address_token", "both"} for reason in reasons)
	if has_name and has_address:
		return "both"
	return "name" if has_name else "address_token"


def generate_candidate_pairs(
	source1_df: pd.DataFrame,
	source2_df: pd.DataFrame,
	source3_df: pd.DataFrame,
	*,
	name_min_length: int = 4,
	min_shared_address_tokens: int = 1,
) -> pd.DataFrame:
	"""Generate only Source1-vs-Source2/3 candidates using country-local indexes.

	The name length and shared-address-token thresholds are configurable. No
	Source2-to-Source3 comparison is performed.
	"""
	if name_min_length < 1:
		raise ValueError("name_min_length must be at least 1")
	if min_shared_address_tokens < 1:
		raise ValueError("min_shared_address_tokens must be at least 1")

	required_columns = {
		"entity_id",
		"normalized_name",
		"address_tokens",
		"is_non_latin_name",
		"normalized_country",
	}
	for source_name, frame in (
		("source1", source1_df),
		("source2", source2_df),
		("source3", source3_df),
	):
		missing = required_columns.difference(frame.columns)
		if missing:
			raise ValueError(f"{source_name}_df is missing columns: {sorted(missing)}")

	source1_records = _record_arrays(source1_df)
	source1_country_positions = _country_positions(source1_df)
	source1_country_counts = {
		country: len(positions) for country, positions in source1_country_positions.items()
	}
	for source_name, frame in (
		("source1", source1_df),
		("source2", source2_df),
		("source3", source3_df),
	):
		counts = frame["normalized_country"].fillna("").value_counts()
		for country, count in counts.items():
			LOGGER.info("Blocking records: source=%s country=%r rows=%d", source_name, country, count)

	candidate_chunks: list[pd.DataFrame] = []
	name_pair_count = 0
	address_pair_count = 0
	for candidate_source, target_frame in (
		("source2", source2_df),
		("source3", source3_df),
	):
		target_records = _record_arrays(target_frame)
		target_country_positions = _country_positions(target_frame)
		for country, target_positions in target_country_positions.items():
			source1_positions = source1_country_positions.get(country)
			source1_count = 0 if source1_positions is None else len(source1_positions)
			if not source1_count or not len(target_positions):
				continue
			country_pairs, country_name_count, country_address_count = _block_country_partition(
				source1_positions,
				target_positions,
				source1_records,
				target_records,
				candidate_source,
				name_min_length,
				min_shared_address_tokens,
			)
			name_pair_count += country_name_count
			address_pair_count += country_address_count
			if not country_pairs.empty:
				candidate_chunks.append(country_pairs)
			LOGGER.info(
				"Blocking partition: candidate_source=%s country=%r "
				"source1_rows=%d target_rows=%d name_pairs=%d address_pairs=%d",
				candidate_source,
				country,
				source1_count,
				len(target_positions),
				country_name_count,
				country_address_count,
			)
		del target_records, target_country_positions

	if candidate_chunks:
		candidate_pairs = pd.concat(candidate_chunks, ignore_index=True)
		pair_columns = ["source1_entity_id", "candidate_entity_id", "candidate_source"]
		if candidate_pairs.duplicated(pair_columns).any():
			candidate_pairs = (
				candidate_pairs.groupby(pair_columns, sort=False, as_index=False)["block_reason"]
				.agg(_combine_block_reasons)
			)
	else:
		candidate_pairs = pd.DataFrame(columns=_OUTPUT_COLUMNS)

	naive_pair_count = len(source1_df) * (len(source2_df) + len(source3_df))
	unique_pair_count = len(candidate_pairs)
	average_per_source1 = unique_pair_count / len(source1_df) if len(source1_df) else 0.0
	reduction_ratio = (
		1.0 - unique_pair_count / naive_pair_count if naive_pair_count else 0.0
	)
	LOGGER.info(
		"Blocking summary: source1=%d source2=%d source3=%d country_partitions=%d "
		"name_pairs=%d address_token_pairs=%d deduplicated_pairs=%d "
		"average_candidates_per_source1=%.4f naive_cross_join=%d reduction_ratio=%.6f",
		len(source1_df),
		len(source2_df),
		len(source3_df),
		len(set(source1_country_counts) | set(source2_df["normalized_country"].fillna("")) | set(source3_df["normalized_country"].fillna(""))),
		name_pair_count,
		address_pair_count,
		unique_pair_count,
		average_per_source1,
		naive_pair_count,
		reduction_ratio,
	)
	return candidate_pairs.loc[:, _OUTPUT_COLUMNS]


def _split_matched_ids(value: object) -> list[str]:
	"""Split a ground-truth match cell into stripped entity identifiers."""
	if value is None or (isinstance(value, float) and np.isnan(value)):
		return []
	if isinstance(value, (list, tuple, set)):
		values = value
	else:
		values = str(value).split(",")
	return [str(item).strip() for item in values if str(item).strip()]


def _truth_pairs(ground_truth_df: pd.DataFrame, source1_df: pd.DataFrame | None) -> pd.DataFrame:
	"""Expand ground truth to pairs, scoped to the supplied Source1 records."""
	required = {"source1_entity_id", "matched_entity_ids"}
	missing = required.difference(ground_truth_df.columns)
	if missing:
		raise ValueError(f"ground_truth_df is missing columns: {sorted(missing)}")
	truth = ground_truth_df.loc[:, ["source1_entity_id", "matched_entity_ids"]]
	if source1_df is not None:
		truth = truth.loc[truth["source1_entity_id"].isin(source1_df["entity_id"])]
	truth = truth.copy()
	truth["matched_entity_ids"] = truth["matched_entity_ids"].map(_split_matched_ids)
	truth = truth.explode("matched_entity_ids").rename(
		columns={"matched_entity_ids": "candidate_entity_id"}
	)
	truth = truth.dropna(subset=["candidate_entity_id"])
	return truth.drop_duplicates(["source1_entity_id", "candidate_entity_id"]).reset_index(drop=True)


def evaluate_blocking_recall(
	candidate_pairs_df: pd.DataFrame,
	ground_truth_df: pd.DataFrame,
	*,
	source1_df: pd.DataFrame | None = None,
	source2_df: pd.DataFrame | None = None,
	source3_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
	"""Measure pair-level blocking recall and report missed Source1 examples.

	Pass the preprocessed source frames to scope a sampled run and calculate the
	non-Latin breakdown plus names/addresses for misses. The two required
	DataFrames alone contain identifiers only, so they cannot provide those
	metadata fields.
	"""
	required_candidate_columns = {"source1_entity_id", "candidate_entity_id"}
	missing = required_candidate_columns.difference(candidate_pairs_df.columns)
	if missing:
		raise ValueError(f"candidate_pairs_df is missing columns: {sorted(missing)}")
	if source1_df is not None and "entity_id" not in source1_df.columns:
		raise ValueError("source1_df must contain entity_id")
	for source_name, frame in (("source2_df", source2_df), ("source3_df", source3_df)):
		if frame is not None and not {"entity_id", "is_non_latin_name"}.issubset(frame.columns):
			raise ValueError(f"{source_name} must contain entity_id and is_non_latin_name")

	true_pairs = _truth_pairs(ground_truth_df, source1_df)
	candidate_keys = candidate_pairs_df.loc[
		:, ["source1_entity_id", "candidate_entity_id"]
	].drop_duplicates()
	recovered_pairs = true_pairs.merge(
		candidate_keys,
		on=["source1_entity_id", "candidate_entity_id"],
		how="inner",
	)
	recovered_pair_set = set(
		zip(recovered_pairs["source1_entity_id"], recovered_pairs["candidate_entity_id"])
	)
	recovered_flags = [
		(source1_id, candidate_id) in recovered_pair_set
		for source1_id, candidate_id in zip(
			true_pairs["source1_entity_id"], true_pairs["candidate_entity_id"]
		)
	]
	true_pairs["recovered"] = recovered_flags

	metadata_available = source1_df is not None and source2_df is not None and source3_df is not None
	if metadata_available:
		source1_metadata = source1_df.drop_duplicates("entity_id", keep="first").set_index("entity_id")
		source1_flags = source1_metadata["is_non_latin_name"].to_dict()
		true_pairs["source1_non_latin"] = true_pairs["source1_entity_id"].map(source1_flags)
		target_identifiers = set(true_pairs["candidate_entity_id"])
		target_flags: dict[object, bool] = {}
		for target_frame in (source2_df, source3_df):
			matched_metadata = target_frame.loc[
				target_frame["entity_id"].isin(target_identifiers),
				["entity_id", "is_non_latin_name"],
			].drop_duplicates("entity_id", keep="first")
			target_flags.update(
				zip(matched_metadata["entity_id"], matched_metadata["is_non_latin_name"])
			)
		true_pairs["candidate_non_latin"] = true_pairs["candidate_entity_id"].map(target_flags)
		source1_known = true_pairs["source1_non_latin"].notna()
		target_known = true_pairs["candidate_non_latin"].notna()
		true_pairs["non_latin_status"] = "unknown"
		true_pairs.loc[
			source1_known
			& target_known
			& (
				true_pairs["source1_non_latin"].astype("boolean")
				| true_pairs["candidate_non_latin"].astype("boolean")
			),
			"non_latin_status",
		] = "involved"
		true_pairs.loc[
			source1_known
			& target_known
			& ~(
				true_pairs["source1_non_latin"].astype("boolean")
				| true_pairs["candidate_non_latin"].astype("boolean")
			),
			"non_latin_status",
		] = "not_involved"
	else:
		true_pairs["non_latin_status"] = "unknown"

	def recall_summary(pairs: pd.DataFrame) -> dict[str, int | float | None]:
		"""Summarize pair recall for a selected set of true matches."""
		total = len(pairs)
		recovered = int(pairs["recovered"].sum()) if total else 0
		return {
			"true_matches": total,
			"recovered_true_matches": recovered,
			"recall_percent": 100.0 * recovered / total if total else None,
		}

	status_summaries = {
		status: recall_summary(true_pairs.loc[true_pairs["non_latin_status"] == status])
		for status in ("involved", "not_involved", "unknown")
	}
	missed_by_source1: dict[object, list[object]] = defaultdict(list)
	for source1_id, candidate_id, recovered in zip(
		true_pairs["source1_entity_id"],
		true_pairs["candidate_entity_id"],
		true_pairs["recovered"],
	):
		if not recovered:
			missed_by_source1[source1_id].append(candidate_id)

	missed_examples: list[dict[str, object]] = []
	if source1_df is not None:
		source1_details = (
			source1_df.drop_duplicates("entity_id", keep="first")
			.set_index("entity_id")
			.reindex(list(missed_by_source1))
		)
	else:
		source1_details = None
	for example_number, (source1_id, missed_ids) in enumerate(missed_by_source1.items()):
		if example_number == 20:
			break
		if source1_details is None:
			normalized_name = None
			normalized_address = None
		else:
			normalized_name = source1_details.iloc[example_number].get("normalized_name")
			normalized_address = source1_details.iloc[example_number].get("normalized_address")
		missed_examples.append(
			{
				"source1_entity_id": source1_id,
				"missed_entity_ids": missed_ids,
				"normalized_name": normalized_name,
				"normalized_address": normalized_address,
			}
		)

	return {
		"evaluated_source1_entities": int(true_pairs["source1_entity_id"].nunique()),
		"true_match_count": len(true_pairs),
		"recovered_true_match_count": len(recovered_pair_set),
		"overall_recall_percent": recall_summary(true_pairs)["recall_percent"],
		"recall_by_non_latin_name": status_summaries,
		"source1_entities_with_missed_matches": len(missed_by_source1),
		"missed_examples": missed_examples,
	}


def _smoke_test() -> None:
	"""Exercise same-country, fallback, country isolation, and source isolation."""
	source1 = pd.DataFrame(
		[
			("S1-latin", "acme widgets limited", ["42", "oak", "springfield"], False, "us"),
			("S1-address", "market house", ["9", "mumbai", "garden"], False, "us"),
			("S1-atlas", "atlas systems", ["77", "harbor", "point"], False, "us"),
			("S1-country", "country test company", ["88", "lake", "view"], False, "us"),
		],
		columns=["entity_id", "normalized_name", "address_tokens", "is_non_latin_name", "normalized_country"],
	)
	source2 = pd.DataFrame(
		[
			("S2-latin", "acme widgets private_limited", ["42", "oak", "springfield"], False, "us"),
			("S2-atlas", "atlas systems llc", ["177", "quiet", "bay"], False, "us"),
			("S2-other-country", "acme widgets limited", ["42", "oak"], False, "france"),
		],
		columns=source1.columns,
	)
	source3 = pd.DataFrame(
		[
			("S3-nonlatin", "भारत उद्योग", ["9", "mumbai", "garden"], True, "us"),
			("S3-atlas", "atlas systems corporation", ["277", "quiet", "bay"], False, "us"),
		],
		columns=source1.columns,
	)
	candidates = generate_candidate_pairs(source1, source2, source3)
	latin_pair = candidates.loc[candidates["candidate_entity_id"] == "S2-latin"].iloc[0]
	fallback_pair = candidates.loc[candidates["candidate_entity_id"] == "S3-nonlatin"].iloc[0]
	assert latin_pair["source1_entity_id"] == "S1-latin"
	assert latin_pair["block_reason"] in {"name", "both"}
	assert fallback_pair["source1_entity_id"] == "S1-address"
	assert fallback_pair["block_reason"] == "address_token"
	assert "S2-other-country" not in set(candidates["candidate_entity_id"])
	assert set(candidates["candidate_source"]) <= {"source2", "source3"}
	assert not candidates["source1_entity_id"].isin({"S2-atlas", "S3-atlas"}).any()
	assert {"S2-atlas", "S3-atlas"}.issubset(set(candidates["candidate_entity_id"]))

	ground_truth = pd.DataFrame(
		{
			"source1_entity_id": ["S1-latin", "S1-address"],
			"matched_entity_ids": ["S2-latin", "S3-nonlatin"],
		}
	)
	print(">>> smoke candidate pairs")
	print(candidates.to_string(index=False))
	print(">>> smoke blocking recall")
	print(
		evaluate_blocking_recall(
			candidates,
			ground_truth,
			source1_df=source1,
			source2_df=source2,
			source3_df=source3,
		)
	)
	print("Smoke test passed: country isolation and Source1-only comparisons verified.")


if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
	_smoke_test()
