"""Ordered contact-filter targets for pick-insert OmniReset."""

from __future__ import annotations


CONTACT_FILTER_TARGETS: list[tuple[str, str]] = [
    ("object", "{ENV_REGEX_NS}/Object"),
    ("receptive", "{ENV_REGEX_NS}/ReceptiveObject"),
    ("table", "{ENV_REGEX_NS}/Table"),
]


def contact_filter_prim_paths() -> list[str]:
    return [expression for _, expression in CONTACT_FILTER_TARGETS]


def object_indices() -> list[int]:
    return [0]


__all__ = ["contact_filter_prim_paths", "object_indices"]
