"""PDF parsing primitives — pypdfium2 wrapper used by the native branch.

Phase 2 ships a thin loader so phase-2 native extraction has a single,
audited surface for talking to ``pypdfium2``. Later phases that need page
rasterisation (phase 4 layout detection) will extend this package.
"""
