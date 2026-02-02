"""
Graph2plan Post-Processing Refinement

Pure Python implementation of geometric refinement for floor plan layouts.
No AI/ML dependencies - uses rule-based geometric algorithms.

Modules:
    geometry_utils: Basic geometric operations
    boundary_align: Snap boxes to boundary walls
    neighbor_align: Align adjacent rooms
    gap_fill: Fill remaining gaps
    refine: Main refinement pipeline
"""

__version__ = "0.1.0"
