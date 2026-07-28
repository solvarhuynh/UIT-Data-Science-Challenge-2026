"""Document-scoped abbreviation extraction and static dictionary access."""

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, Mapping, Optional

from udsc2026.ingestion.cleaners.patterns import ABBREVIATION_DEFINITION


DEFAULT_ABBREVIATIONS_PATH = (
    Path(__file__).resolve().parents[1] / "legal_structure" / "abbreviations.json"
)


def extract_abbreviations(text: str) -> Dict[str, str]:
    """Extract definitions such as ``Bộ luật ... (sau đây gọi là BLLĐ)``."""
    mapping: Dict[str, str] = {}
    for full_form, abbreviation in ABBREVIATION_DEFINITION.findall(text):
        normalized_abbreviation = abbreviation.strip().strip(".,;:")
        normalized_full_form = full_form.strip().rstrip(",;.")
        if normalized_abbreviation and normalized_full_form:
            mapping[normalized_abbreviation] = normalized_full_form
    return mapping


@lru_cache(maxsize=1)
def load_static_abbreviations(
    path: Optional[Path] = None,
) -> Dict[str, str]:
    """Load common legal abbreviations once, returning an empty mapping if absent."""
    dictionary_path = path or DEFAULT_ABBREVIATIONS_PATH
    if not dictionary_path.exists():
        return {}
    with dictionary_path.open("r", encoding="utf-8") as source_file:
        loaded = json.load(source_file)
    if not isinstance(loaded, dict):
        raise ValueError("Static abbreviation dictionary must be a JSON object.")
    return {str(key): str(value) for key, value in loaded.items()}


def build_abbreviation_dictionary(
    text: str,
    static_abbreviations: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    """Combine static terms with document-specific definitions.

    Definitions in the current document override common static meanings because
    the same abbreviation can have different legal meanings between documents.
    """
    dictionary = dict(static_abbreviations or load_static_abbreviations())
    dictionary.update(extract_abbreviations(text))
    return dictionary


def expanded_terms_in_text(
    text: str,
    abbreviations: Mapping[str, str],
) -> Dict[str, str]:
    """Return only abbreviations present in text, without changing its wording."""
    return {
        abbreviation: full_form
        for abbreviation, full_form in abbreviations.items()
        if re.search(r"(?<!\w){0}(?!\w)".format(re.escape(abbreviation)), text)
    }
