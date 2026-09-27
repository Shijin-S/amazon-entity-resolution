"""Country-aware candidate generation and blocking recall validation."""

import logging
import gc
import heapq
import shutil
from array import array
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from tempfile import mkdtemp
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import pandas as pd
import psutil
import pyarrow.dataset as ds
import pyarrow as pa
import pyarrow.parquet as pq


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


_BLOCKING_COLUMNS = ["entity_id", "normalized_name", "address_tokens", "is_non_latin_name"]


class MemorySafetyLimitExceeded(RuntimeError):
	"""Raised when blocking reaches its configured RSS safety limit."""

	def __init__(self, message: str, candidate_output_path: str):
		super().__init__(message)
		self.candidate_output_path = candidate_output_path


class _CandidateBatchWriter:
	"""Persist candidate batches independently so partial output survives errors."""

	def __init__(self, output_path: str):
		self.output_path = Path(output_path)
		self.output_path.mkdir(parents=True, exist_ok=True)
		self.batch_count = 0
		self.row_count = 0
		self.parquet_writer: pq.ParquetWriter | None = None

	def write(self, candidates: pd.DataFrame) -> None:
		"""Write one non-empty batch and release it from the caller immediately."""
		if candidates.empty:
			return
		table = pa.Table.from_pandas(candidates, preserve_index=False)
		if self.parquet_writer is None:
			self.parquet_writer = pq.ParquetWriter(
				self.output_path / "candidates.parquet", table.schema
			)
		self.parquet_writer.write_table(table, row_group_size=len(candidates))
		self.batch_count += 1
		self.row_count += len(candidates)

	def flush(self) -> None:
		"""Close the Parquet writer so all completed row groups are readable."""
		if self.parquet_writer is not None:
			self.parquet_writer.close()
			self.parquet_writer = None
		LOGGER.info(
			"Candidate output checkpoint flushed: path=%s batches=%d rows=%d",
			self.output_path,
			self.batch_count,
			self.row_count,
		)


def _check_memory_safety(
	process: psutil.Process,
	limit_mb: int,
	writer: _CandidateBatchWriter,
	country: str,
	phase: str,
	batch_number: int,
) -> float:
	"""Raise before processing a batch when process RSS exceeds the safety limit."""
	rss_mb = process.memory_info().rss / 1024 / 1024
	if rss_mb > limit_mb:
		message = (
			"Memory safety limit exceeded: "
			f"country={country!r} phase={phase} batch={batch_number} "
			f"rss_mb={rss_mb:.1f} limit_mb={limit_mb} "
			f"candidate_output={writer.output_path} "
			f"saved_candidate_rows={writer.row_count}"
		)
		LOGGER.critical(message)
		writer.flush()
		raise MemorySafetyLimitExceeded(message, str(writer.output_path))
	return rss_mb


def _log_index_diagnostics(
	country: str,
	candidate_source: str,
	name_index: dict[tuple[str, str], array],
	address_index: dict[str, list[array]],
	dropped_name_keys: set[tuple[str, str]],
	dropped_address_tokens: set[str],
) -> None:
	"""Log index size, most common postings, and high-frequency key counts."""
	thresholds = (1_000, 10_000, 100_000)
	name_counts = dict.fromkeys(thresholds, 0)
	address_counts = dict.fromkeys(thresholds, 0)
	top_names = heapq.nlargest(
		20,
		((key, len(postings)) for key, postings in name_index.items()),
		key=lambda item: item[1],
	)
	top_addresses = heapq.nlargest(
		20,
		((token, len(postings[0])) for token, postings in address_index.items()),
		key=lambda item: item[1],
	)
	for postings in name_index.values():
		length = len(postings)
		for threshold in thresholds:
			name_counts[threshold] += length > threshold
	for postings in address_index.values():
		length = len(postings[0])
		for threshold in thresholds:
			address_counts[threshold] += length > threshold
	LOGGER.info(
		"Index diagnostics: country=%r source=%s unique_name_keys=%d "
		"name_postings_gt_1000=%d gt_10000=%d gt_100000=%d "
		"top_name_keys=%s dropped_name_keys=%d",
		country,
		candidate_source,
		len(name_index),
		name_counts[1_000],
		name_counts[10_000],
		name_counts[100_000],
		top_names,
		len(dropped_name_keys),
	)
	LOGGER.info(
		"Index diagnostics: country=%r source=%s unique_address_tokens=%d "
		"address_postings_gt_1000=%d gt_10000=%d gt_100000=%d "
		"top_address_tokens=%s dropped_address_tokens=%d",
		country,
		candidate_source,
		len(address_index),
		address_counts[1_000],
		address_counts[10_000],
		address_counts[100_000],
		top_addresses,
		len(dropped_address_tokens),
	)


def _country_filter(country: str) -> ds.Expression:
	country_field = ds.field("normalized_country")
	if country == "":
		return (country_field == "") | country_field.is_null()
	return country_field == country


def _iter_country_batches(
	path: str,
	country: str,
	columns: list[str],
	batch_size: int,
) -> Iterable[pd.DataFrame]:
	dataset = ds.dataset(path, format="parquet")
	scanner = dataset.scanner(
		columns=columns,
		filter=_country_filter(country),
		batch_size=batch_size,
		batch_readahead=1,
		fragment_readahead=1,
		use_threads=False,
	)
	for record_batch in scanner.to_batches():
		yield record_batch.to_pandas()


def _build_target_index(
	path: str,
	country: str,
	candidate_source: str,
	batch_size: int,
	name_min_length: int,
	max_postings_per_token: int,
	memory_safety_limit_mb: int,
	process: psutil.Process,
	writer: _CandidateBatchWriter,
) -> tuple[
	list[object],
	list[tuple[str, ...]],
	dict[tuple[str, str], array],
	dict[str, list[array]],
	set[tuple[str, str]],
]:
	"""Incrementally build compact blocking indexes for one target source."""
	target_ids: list[object] = []
	target_address_tokens: list[tuple[str, ...]] = []
	name_index: dict[tuple[str, str], array] = {}
	address_index: dict[str, list[array]] = {}
	dropped_name_keys: set[tuple[str, str]] = set()
	dropped_address_tokens: set[str] = set()
	for batch_number, batch_frame in enumerate(
		_iter_country_batches(path, country, _BLOCKING_COLUMNS, batch_size), start=1
	):
		rss_mb = _check_memory_safety(
			process,
			memory_safety_limit_mb,
			writer,
			country,
			"target_index",
			batch_number,
		)
		LOGGER.info(
			"Starting blocking batch: country=%r source=%s phase=target_index "
			"batch=%d rows=%d rss_mb=%.1f",
			country,
			candidate_source,
			batch_number,
			len(batch_frame),
			rss_mb,
		)
		for entity_id, name, address_tokens, is_non_latin in batch_frame.itertuples(
			index=False, name=None
		):
			target_position = len(target_ids)
			target_ids.append(entity_id)
			fallback = _needs_address_fallback(name, is_non_latin, name_min_length)
			name_key = None if is_non_latin else _name_block_key(name, name_min_length)
			tokens = _filtered_address_tokens(address_tokens)
			target_address_tokens.append(tuple(tokens) if name_key is not None else ())
			if name_key is not None and name_key not in dropped_name_keys:
				name_postings = name_index.get(name_key)
				original_length = 1 if name_postings is None else len(name_postings) + 1
				if original_length > max_postings_per_token:
					name_index.pop(name_key, None)
					dropped_name_keys.add(name_key)
					LOGGER.warning(
						"Dropped name block key %r: original posting-list length=%d "
						"exceeds max_postings_per_token=%d",
						name_key,
						original_length,
						max_postings_per_token,
					)
				else:
					if name_postings is None:
						name_postings = array("I")
						name_index[name_key] = name_postings
					name_postings.append(target_position)

			for token in tokens:
				if token in dropped_address_tokens:
					continue
				postings = address_index.get(token)
				original_length = 1 if postings is None else len(postings[0]) + 1
				if original_length > max_postings_per_token:
					address_index.pop(token, None)
					dropped_address_tokens.add(token)
					LOGGER.warning(
						"Dropped address token %r: original posting-list length=%d "
						"exceeds max_postings_per_token=%d",
						token,
						original_length,
						max_postings_per_token,
					)
					continue
				if postings is None:
					postings = [array("I"), array("I")]
					address_index[token] = postings
				postings[0].append(target_position)
				if fallback:
					postings[1].append(target_position)
		del batch_frame
		gc.collect()
	_log_index_diagnostics(
		country,
		candidate_source,
		name_index,
		address_index,
		dropped_name_keys,
		dropped_address_tokens,
	)
	return target_ids, target_address_tokens, name_index, address_index, dropped_name_keys


def _block_source1_batch(
	source1_frame: pd.DataFrame,
	target_ids: list[object],
	target_address_tokens: list[tuple[str, ...]],
	name_index: dict[tuple[str, str], array],
	address_index: dict[str, list[array]],
	dropped_name_keys: set[tuple[str, str]],
	candidate_source: str,
	name_min_length: int,
	min_shared_address_tokens: int,
) -> tuple[pd.DataFrame, int, int]:
	"""Match one Source1 batch against a complete country-local target index."""
	pairs: list[tuple[object, object, str, str]] = []
	name_pair_count = 0
	address_pair_count = 0
	for source1_id, name, address_tokens, is_non_latin in source1_frame[
		["entity_id", "normalized_name", "address_tokens", "is_non_latin_name"]
	].itertuples(index=False, name=None):
		address_tokens_set = _filtered_address_tokens(address_tokens)
		name_key = None if is_non_latin else _name_block_key(name, name_min_length)
		name_hits = list(name_index.get(name_key, ())) if name_key is not None else []
		name_pair_count += len(name_hits)

		name_address_hits = {
			target_position
			for target_position in name_hits
			if len(address_tokens_set.intersection(target_address_tokens[target_position]))
			>= min_shared_address_tokens
		}

		fallback = _needs_address_fallback(name, is_non_latin, name_min_length) or (
			name_key is not None and name_key in dropped_name_keys
		)
		posting_position = 0 if fallback else 1
		address_hit_counts: dict[int, int] = defaultdict(int)
		for token in address_tokens_set:
			postings = address_index.get(token)
			if postings is not None:
				for target_position in postings[posting_position]:
					address_hit_counts[target_position] += 1

		fallback_address_hits = {
			target_position
			for target_position, shared_tokens in address_hit_counts.items()
			if shared_tokens >= min_shared_address_tokens
		}
		address_hits = name_address_hits | fallback_address_hits
		address_pair_count += len(address_hits)
		name_hit_set = set(name_hits)
		for target_position in sorted(name_hit_set | address_hits):
			if target_position in name_hit_set and target_position in address_hits:
				reason = "both"
			elif target_position in name_hit_set:
				reason = "name"
			else:
				reason = "address_token"
			pairs.append((source1_id, target_ids[target_position], candidate_source, reason))

	return pd.DataFrame(pairs, columns=_OUTPUT_COLUMNS), name_pair_count, address_pair_count


def _combine_block_reasons(reasons: Iterable[str]) -> str:
	"""Combine duplicate pair reasons into one stable strategy label."""
	has_name = any(reason in {"name", "both"} for reason in reasons)
	has_address = any(reason in {"address_token", "both"} for reason in reasons)
	if has_name and has_address:
		return "both"
	return "name" if has_name else "address_token"


def generate_candidate_pairs(
	source1_path: str,
	source2_path: str,
	source3_path: str,
	*,
	name_min_length: int = 4,
	min_shared_address_tokens: int = 1,
	batch_size: int = 300_000,
	match_batch_size: int = 64,
	max_postings_per_token: int = 2_000,
	memory_safety_limit_mb: int = 9_000,
	countries: list[str] | None = None,
	candidate_output_dir: str | None = None,
	return_candidate_dataframe: bool = False,
) -> pd.DataFrame | str:
	"""Generate only Source1-vs-Source2/3 candidates using country-local indexes.

	The thresholds and batch sizes are configurable. Every candidate batch is
	written to Parquet immediately. Set ``return_candidate_dataframe=False`` to
	return the output directory path without loading all candidates into memory.
	Optionally restrict processing to selected countries. No Source2-to-Source3
	comparison is performed.
	"""
	if name_min_length < 1:
		raise ValueError("name_min_length must be at least 1")
	if min_shared_address_tokens < 1:
		raise ValueError("min_shared_address_tokens must be at least 1")
	if batch_size < 1:
		raise ValueError("batch_size must be at least 1")
	if match_batch_size < 1:
		raise ValueError("match_batch_size must be at least 1")
	if max_postings_per_token < 1:
		raise ValueError("max_postings_per_token must be at least 1")
	if memory_safety_limit_mb < 1:
		raise ValueError("memory_safety_limit_mb must be at least 1")

	if candidate_output_dir is None:
		output_path = mkdtemp(prefix="entity-resolution-candidates-")
	else:
		output_root = Path(candidate_output_dir)
		output_root.mkdir(parents=True, exist_ok=True)
		output_path = mkdtemp(prefix="run-", dir=output_root)
	writer = _CandidateBatchWriter(output_path)
	name_pair_count = 0
	address_pair_count = 0
	process = psutil.Process()
	LOGGER.info("Blocking function entry: rss_mb=%.1f", process.memory_info().rss / 1024 / 1024)
	source1_country_column = pd.read_parquet(source1_path, columns=["normalized_country"])[
		"normalized_country"
	].fillna("")
	source1_country_counts = source1_country_column.value_counts().to_dict()
	source1_countries = list(source1_country_counts)
	del source1_country_column
	gc.collect()
	if countries is not None:
		requested_countries = set(countries)
		source1_countries = [
			country for country in source1_countries if country in requested_countries
		]
	source1_count = sum(source1_country_counts[country] for country in source1_countries)
	source2_count = 0
	source3_count = 0
	country_partition_count = len(source1_countries)
	for country in source1_countries:
		LOGGER.info(
			"Starting country partition: country=%r source1_rows=%d rss_mb=%.1f",
			country,
			source1_country_counts[country],
			process.memory_info().rss / 1024 / 1024,
		)
		for candidate_source, target_path in (
			("source2", source2_path),
			("source3", source3_path),
		):
			(
				target_ids,
				target_address_tokens,
				name_index,
				address_index,
				dropped_name_keys,
			) = _build_target_index(
				target_path,
				country,
				candidate_source,
				batch_size,
				name_min_length,
				max_postings_per_token,
				memory_safety_limit_mb,
				process,
				writer,
			)
			if candidate_source == "source2":
				source2_count += len(target_ids)
			else:
				source3_count += len(target_ids)
			for batch_number, source1_batch in enumerate(
				_iter_country_batches(
					source1_path,
					country,
					_BLOCKING_COLUMNS,
					min(batch_size, match_batch_size),
				),
				start=1,
			):
				rss_mb = _check_memory_safety(
					process,
					memory_safety_limit_mb,
					writer,
					country,
					"source1_match",
					batch_number,
				)
				LOGGER.info(
					"Starting blocking batch: country=%r source=%s phase=source1_match "
					"batch=%d rows=%d rss_mb=%.1f",
					country,
					candidate_source,
					batch_number,
					len(source1_batch),
					rss_mb,
				)
				country_pairs, country_name_count, country_address_count = _block_source1_batch(
					source1_batch,
					target_ids,
					target_address_tokens,
					name_index,
					address_index,
					dropped_name_keys,
					candidate_source,
					name_min_length,
					min_shared_address_tokens,
				)
				name_pair_count += country_name_count
				address_pair_count += country_address_count
				writer.write(country_pairs)
				del source1_batch, country_pairs
				gc.collect()
				_check_memory_safety(
					process,
					memory_safety_limit_mb,
					writer,
					country,
					"source1_match_after_batch",
					batch_number,
				)
			del target_ids, target_address_tokens, name_index, address_index, dropped_name_keys
			gc.collect()
		LOGGER.info(
			"Finished country partition: country=%r rss_mb=%.1f",
			country,
			process.memory_info().rss / 1024 / 1024,
		)
	del source1_countries

	writer.flush()
	unique_pair_count = writer.row_count
	if return_candidate_dataframe:
		candidate_pairs = (
			pd.read_parquet(writer.output_path)
			if writer.batch_count
			else pd.DataFrame(columns=_OUTPUT_COLUMNS)
		)
		pair_columns = ["source1_entity_id", "candidate_entity_id", "candidate_source"]
		if candidate_pairs.duplicated(pair_columns).any():
			candidate_pairs = (
				candidate_pairs.groupby(pair_columns, sort=False, as_index=False)["block_reason"]
				.agg(_combine_block_reasons)
			)
		unique_pair_count = len(candidate_pairs)
		shutil.rmtree(writer.output_path)

	naive_pair_count = source1_count * (source2_count + source3_count)
	average_per_source1 = unique_pair_count / source1_count if source1_count else 0.0
	reduction_ratio = (
		1.0 - unique_pair_count / naive_pair_count if naive_pair_count else 0.0
	)
	LOGGER.info(
		"Blocking summary: source1=%d source2=%d source3=%d country_partitions=%d "
		"name_pairs=%d address_token_pairs=%d deduplicated_pairs=%d "
		"average_candidates_per_source1=%.4f naive_cross_join=%d reduction_ratio=%.6f",
		source1_count,
		source2_count,
		source3_count,
		country_partition_count,
		name_pair_count,
		address_pair_count,
		unique_pair_count,
		average_per_source1,
		naive_pair_count,
		reduction_ratio,
	)
	if return_candidate_dataframe:
		return candidate_pairs.loc[:, _OUTPUT_COLUMNS]
	return str(writer.output_path)


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


def _load_recall_metadata(
	source1_path: str,
	source2_path: str,
	source3_path: str,
	source1_ids: set[object],
	target_ids: set[object],
	batch_size: int = 300_000,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	"""Load only metadata rows referenced by recall, one country at a time."""
	source1_chunks: list[pd.DataFrame] = []
	source2_chunks: list[pd.DataFrame] = []
	source3_chunks: list[pd.DataFrame] = []
	countries = (
		pd.read_parquet(source1_path, columns=["normalized_country"])
		["normalized_country"]
		.fillna("")
		.drop_duplicates()
		.tolist()
	)
	for country in countries:
		for batch in _iter_country_batches(
			source1_path,
			country,
			["entity_id", "normalized_name", "is_non_latin_name"],
			batch_size,
		):
			matched = batch.loc[batch["entity_id"].isin(source1_ids)]
			if not matched.empty:
				source1_chunks.append(matched.copy())
			del batch, matched
		for path, chunks in ((source2_path, source2_chunks), (source3_path, source3_chunks)):
			for batch in _iter_country_batches(
				path, country, ["entity_id", "is_non_latin_name"], batch_size
			):
				matched = batch.loc[batch["entity_id"].isin(target_ids)]
				if not matched.empty:
					chunks.append(matched.copy())
				del batch, matched
		gc.collect()

	def combine(chunks: list[pd.DataFrame], columns: list[str]) -> pd.DataFrame:
		return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=columns)

	return (
		combine(source1_chunks, ["entity_id", "normalized_name", "is_non_latin_name"]),
		combine(source2_chunks, ["entity_id", "is_non_latin_name"]),
		combine(source3_chunks, ["entity_id", "is_non_latin_name"]),
	)


def evaluate_blocking_recall(
	candidate_pairs_df: pd.DataFrame,
	ground_truth_df: pd.DataFrame,
	*,
	source1_path: str | None = None,
	source2_path: str | None = None,
	source3_path: str | None = None,
		batch_size: int = 300_000,
) -> dict[str, Any]:
	"""Measure pair-level blocking recall and report missed Source1 examples.

	Pass the preprocessed source paths to calculate the non-Latin breakdown plus
	names for misses. Recall metadata is loaded one country at a time and only
	rows referenced by the ground truth or candidate pairs are retained.
	"""
	required_candidate_columns = {"source1_entity_id", "candidate_entity_id"}
	missing = required_candidate_columns.difference(candidate_pairs_df.columns)
	if missing:
		raise ValueError(f"candidate_pairs_df is missing columns: {sorted(missing)}")
	paths = (source1_path, source2_path, source3_path)
	if any(path is None for path in paths) and any(path is not None for path in paths):
		raise ValueError("source1_path, source2_path, and source3_path must be supplied together")
	if batch_size < 1:
		raise ValueError("batch_size must be at least 1")
	metadata_available = all(path is not None for path in paths)
	preliminary_truth = _truth_pairs(ground_truth_df, None)
	if metadata_available:
		source1_df, source2_df, source3_df = _load_recall_metadata(
			source1_path,
			source2_path,
			source3_path,
			set(preliminary_truth["source1_entity_id"]),
			set(preliminary_truth["candidate_entity_id"]),
			batch_size,
		)
	else:
		source1_df = source2_df = source3_df = None
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
	ground_truth = pd.DataFrame(
		{
			"source1_entity_id": ["S1-latin", "S1-address"],
			"matched_entity_ids": ["S2-latin", "S3-nonlatin"],
		}
	)
	with TemporaryDirectory() as temp_dir:
		paths = []
		for source_name, frame in (("source1", source1), ("source2", source2), ("source3", source3)):
			path = f"{temp_dir}/{source_name}.parquet"
			frame.to_parquet(path, index=False)
			paths.append(path)
		candidates = generate_candidate_pairs(*paths, return_candidate_dataframe=True)
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

		print(">>> smoke candidate pairs")
		print(candidates.to_string(index=False))
		print(">>> smoke blocking recall")
		print(
			evaluate_blocking_recall(
				candidates,
				ground_truth,
				source1_path=paths[0],
				source2_path=paths[1],
				source3_path=paths[2],
			)
		)
	print("Smoke test passed: country isolation and Source1-only comparisons verified.")


if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
	_smoke_test()
