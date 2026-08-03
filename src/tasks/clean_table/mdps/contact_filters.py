from __future__ import annotations


# The list order is the ContactSensor force-matrix filter order. Object contact must
# remain index zero because the frozen policy and shared grasp terms rely on it.
CONTACT_FILTER_TARGETS: list[tuple[str, str]] = [
    ("object", "{ENV_REGEX_NS}/Object/.*"),
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
    object_index = CONTACT_TARGET_INDEX["object"]
    return [index for index in range(len(CONTACT_FILTER_TARGETS)) if index != object_index]
