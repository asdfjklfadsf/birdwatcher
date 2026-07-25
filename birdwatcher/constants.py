"""Shared runtime constants for Bird Watcher."""

# Multiplier applied to species that are merely plausible for the region, as
# opposed to locally preferred. Clamped to REGIONAL_PRIOR_WEIGHT at use time so
# it can never exceed the preferred multiplier.
BROAD_REGIONAL_PRIOR_WEIGHT = 1.5

# Maximum number of cached BioCLIP species-prompt embedding tables. The
# candidate set varies per frame, so this must stay bounded.
TEXT_FEATURE_CACHE_SIZE = 32

# Boxes overlapping more than this are treated as one bird when the
# low-confidence sweep produces duplicates from overlapping passes.
DUPLICATE_DETECTION_IOU = 0.5
