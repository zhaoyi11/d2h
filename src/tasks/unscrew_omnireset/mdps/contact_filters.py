"""Ordered contact-filter targets for unscrew OmniReset."""

from __future__ import annotations


CONTACT_FILTER_TARGETS: list[tuple[str, str]] = [
    ("object", "{ENV_REGEX_NS}/Object"),
    ("receptive", "{ENV_REGEX_NS}/ReceptiveObject"),
    ("table", "{ENV_REGEX_NS}/Table"),
]
CONTACT_TARGET_INDEX = {
    name: index for index, (name, _) in enumerate(CONTACT_FILTER_TARGETS)
}


def contact_filter_prim_paths() -> list[str]:
    return [expression for _, expression in CONTACT_FILTER_TARGETS]


def object_indices() -> list[int]:
    return [CONTACT_TARGET_INDEX["object"]]


def external_indices() -> list[int]:
    return [
        index for name, index in CONTACT_TARGET_INDEX.items() if name != "object"
    ]


__all__ = ["contact_filter_prim_paths", "external_indices", "object_indices"]
