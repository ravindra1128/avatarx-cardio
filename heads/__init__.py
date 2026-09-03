"""Endpoint heads (v0.2 L4). Importing this package registers the
built-in heads; `heads.base.enabled_heads(cfg)` is the wiring surface."""
from heads.base import (EndpointHead, HeadResult, HeadRegistryError,   # noqa
                        register_head, get_head, available_heads,
                        enabled_heads)
import heads.head_afib          # noqa: F401  (registration side-effect)
import heads.head_rate_flags    # noqa: F401
import heads.head_rhythm_map    # noqa: F401
import heads.head_flutter        # noqa: F401  (v0.6 §F-gated; supersedes
                                 # the v0.2 head_flutter_suspicion stub)
import heads.head_burden        # noqa: F401  (internal-only)
import heads.head_irregularity  # noqa: F401  (research-only, v0.3 T6)
import heads.head_recovery      # noqa: F401  (v0.4 session-only, MEASURED)
import heads.head_fitness       # noqa: F401  (v0.4 §V-gated, disabled)
import heads.head_trend         # noqa: F401  (v0.4 §V-gated, disabled)
import heads.head_vascular      # noqa: F401  (research-flagged, V-a inert)
import heads.head_vasotone      # noqa: F401  (research-flagged, W-a inert)
import heads.head_regularity    # noqa: F401  (v0.7 §R-gated substrate head)
