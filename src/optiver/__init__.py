"""Optiver — Trading at the Close. Phase 1: harness and honest floor.

Import order matters only in that `config` holds the paths every other module
resolves against; nothing here has a side effect on import.
"""

__all__ = ["config", "seeding", "data", "splits", "features", "evaluate", "baselines"]
