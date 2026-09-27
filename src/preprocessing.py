"""Data preprocessing and attribute normalization for raw entity records."""

import re
import unicodedata

import pandas as pd


_ADDRESS_ABBREVIATIONS = {
	"st": "street",
	"rd": "road",
	"ave": "avenue",
	"blvd": "boulevard",
	"ln": "lane",
	"dr": "drive",
	"hwy": "highway",
	"pkwy": "parkway",
	"apt": "apartment",
	"ste": "suite",
	"fl": "floor",
}


def _coerce_text(value: object) -> str:
	"""Return a safe string representation, using empty text for nulls."""
	if value is None:
		return ""
	try:
		if pd.isna(value):
			return ""
	except (TypeError, ValueError):
		pass
	return value if isinstance(value, str) else str(value)


def _strip_punctuation(text: str) -> str:
	"""Replace punctuation with spaces while preserving Unicode letters/marks."""
	return "".join(
		character
		if character.isspace()
		or unicodedata.category(character).startswith(("L", "M", "N"))
		else " "
		for character in text
	)


def normalize_unicode(text: str) -> str:
	"""NFKD-normalize text and remove combining marks attached to Latin letters.

	Null-like input is returned as an empty string. Non-Latin characters and
	their combining marks are retained; this function does not transliterate.
	"""
	normalized = unicodedata.normalize("NFKD", _coerce_text(text))
	result: list[str] = []
	previous_base_is_latin = False
	for character in normalized:
		if unicodedata.combining(character):
			if not previous_base_is_latin:
				result.append(character)
			continue
		result.append(character)
		if unicodedata.category(character).startswith("L"):
			previous_base_is_latin = unicodedata.name(character, "").startswith("LATIN ")
		else:
			previous_base_is_latin = False
	return "".join(result)


def is_non_latin_script(text: str) -> bool:
	"""Return whether text contains a letter outside the Latin script."""
	for character in _coerce_text(text):
		if unicodedata.category(character).startswith("L") and not unicodedata.name(
			character, ""
		).startswith("LATIN "):
			return True
	return False


def normalize_business_name(name: str) -> str:
	"""Lowercase and clean a business name while retaining canonical suffixes."""
	normalized = normalize_unicode(name).lower()
	normalized = _strip_punctuation(normalized)
	normalized = re.sub(r"\s+", " ", normalized).strip()
	normalized = re.sub(
		r"\b(?:pvt|private)\s+(?:ltd|limited)\b", "private_limited", normalized
	)
	normalized = re.sub(r"\b(?:corp|corporation)\b", "corporation", normalized)
	normalized = re.sub(r"\b(?:ltd|limited)\b", "limited", normalized)
	return re.sub(r"\s+", " ", normalized).strip()


def normalize_address(address: str) -> str:
	"""Lowercase and clean an address; null-like addresses become empty strings."""
	normalized = normalize_unicode(address).lower()
	normalized = _strip_punctuation(normalized)
	tokens = normalized.split()
	return " ".join(_ADDRESS_ABBREVIATIONS.get(token, token) for token in tokens)


def extract_address_tokens(address: str) -> list[str]:
	"""Extract numeric and at-least-three-letter tokens from an address."""
	normalized = normalize_address(address)
	tokens: list[str] = []
	current: list[str] = []
	current_kind: str | None = None
	letter_count = 0

	def finish_token() -> None:
		"""Append the current token when it meets its type-specific threshold."""
		if current and (current_kind == "number" or letter_count >= 3):
			tokens.append("".join(current))

	for character in normalized:
		category = unicodedata.category(character)
		if category.startswith("L"):
			kind = "letter"
		elif category.startswith("M") and current_kind == "letter":
			kind = "letter"
		elif category.startswith("N"):
			kind = "number"
		else:
			kind = None

		if kind != current_kind:
			finish_token()
			current = []
			letter_count = 0
			current_kind = kind
		if kind is not None:
			current.append(character)
			if kind == "letter" and category.startswith("L"):
				letter_count += 1
	finish_token()
	return tokens


def normalize_country(country: str) -> str:
	"""Lowercase and trim a country label without restricting its vocabulary."""
	return _coerce_text(country).strip().lower()


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
	"""Return a copy of a source frame with normalized matching attributes."""
	result = df.copy()

	names = result.get("business_name", pd.Series(index=result.index, dtype=object))
	addresses = result.get("business_address", pd.Series(index=result.index, dtype=object))
	countries = result.get("country", pd.Series(index=result.index, dtype=object))

	result["normalized_name"] = names.map(normalize_business_name)
	result["normalized_address"] = addresses.map(normalize_address)
	result["address_tokens"] = addresses.map(extract_address_tokens)
	result["is_non_latin_name"] = names.map(is_non_latin_script).astype(bool)
	result["normalized_country"] = countries.map(normalize_country)
	return result
