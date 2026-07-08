from __future__ import annotations

# ---------------------------------------------------------------------------
# Contact-sensing target registry (cupcake_on_plate)
# ---------------------------------------------------------------------------
# Ordered list of prim-path expressions that the fingertip contact sensors filter
# against. The list *position* equals the filter index in
# ``ContactSensor.data.force_matrix_w[:, :, k, :]``. This is the single source of
# truth: the HRL scene builds every fingertip sensor's ``filter_prim_paths_expr``
# from it (see env_cfg.py), and the low-level contact observations reference targets
# by *role* (object vs. external) resolved through this registry -- never by a
# literal index.
#
# IMPORTANT: "object" must stay first (index 0) so the object-grasp contact terms
# keep reading object contact only. The external targets are the surfaces the hand
# can brace/pivot against while flipping and placing the cupcake: the plate
# (receptacle) and the table slab. Adding another external target is a one-line
# edit here plus a matching scene asset.
CONTACT_FILTER_TARGETS: list[tuple[str, str]] = [
    ("object",    "{ENV_REGEX_NS}/Object"),
    ("receptive", "{ENV_REGEX_NS}/ReceptiveObject"),
    ("table",     "{ENV_REGEX_NS}/Table"),
]
CONTACT_TARGET_INDEX: dict[str, int] = {
    name: i for i, (name, _) in enumerate(CONTACT_FILTER_TARGETS)
}


def contact_filter_prim_paths() -> list[str]:
    """Prim-path expressions for the fingertip sensors' ``filter_prim_paths_expr``."""
    return [expr for _, expr in CONTACT_FILTER_TARGETS]


def object_indices() -> list[int]:
    """Filter indices that correspond to the manipulated object (index 0)."""
    return [CONTACT_TARGET_INDEX["object"]]


def external_indices() -> list[int]:
    """Filter indices of every non-object contact target (plate, table, ...).

    Returns an empty list when no external targets are registered.
    """
    obj = CONTACT_TARGET_INDEX["object"]
    return [i for i in range(len(CONTACT_FILTER_TARGETS)) if i != obj]
