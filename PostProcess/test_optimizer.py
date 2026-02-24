"""
Quick test for the CP-SAT optimizer.
No model or data files required — uses synthetic overlapping rooms.

Run from PostProcess/:
    py -3.11 test_optimizer.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from optimizer.solver import optimize_layout

# ---------------------------------------------------------------------------
# Synthetic layout: 5 rooms, deliberately overlapping
# Format: [x0, y0, x1, y1]  in 256x256 pixel space
# ---------------------------------------------------------------------------
# Imagine a small flat:
#   LivingRoom (0) | Kitchen (2)
#   ----------------+------------
#   Bathroom (3)   | Bedroom (1)
#                  | Bedroom2 (7)

boxes = np.array([
    [ 10,  10,  120, 130],   # 0  LivingRoom  (top-left)
    [100,  10,  200, 130],   # 1  MasterRoom  (top-right) — overlaps LivingRoom on x
    [ 10, 110,   80, 200],   # 2  Kitchen     (bottom-left) — overlaps LivingRoom on y
    [ 70, 100,  160, 200],   # 3  Bathroom    (bottom-mid)  — overlaps multiple
    [150,  60,  230, 200],   # 4  DiningRoom  (right)
], dtype=int)

types = np.array([0, 1, 2, 3, 4], dtype=int)

# Edges: [room_u, room_v, predicate]
# Predicates: 0=left-above, 1=left-below, 2=left-of, 3=above,
#             4=inside, 5=surrounding, 6=below, 7=right-of, 8=right-above, 9=right-below
edges = np.array([
    [0, 1, 7],   # LivingRoom right-of MasterRoom  (0 is LEFT of 1 → predicate 2)
    [0, 2, 6],   # LivingRoom below Kitchen        (0 is ABOVE 2 → predicate 3)
    [1, 4, 6],   # MasterRoom above DiningRoom     (4 is BELOW 1 → wait let's use simpler)
    [2, 3, 7],   # Kitchen right-of Bathroom
], dtype=int)

# Boundary: simple rectangle 0..240 x 0..240
boundary = np.array([
    [  0,   0, 0, 0],
    [240,   0, 0, 0],
    [240, 240, 0, 0],
    [  0, 240, 0, 0],
], dtype=int)

# ---------------------------------------------------------------------------
# Print initial state
# ---------------------------------------------------------------------------
def check_overlaps(boxes):
    overlaps = []
    K = len(boxes)
    for i in range(K):
        for j in range(i+1, K):
            ax0, ay0, ax1, ay1 = boxes[i]
            bx0, by0, bx1, by1 = boxes[j]
            if ax0 < bx1 and ax1 > bx0 and ay0 < by1 and ay1 > by0:
                overlap_w = min(ax1, bx1) - max(ax0, bx0)
                overlap_h = min(ay1, by1) - max(ay0, by0)
                overlaps.append((i, j, overlap_w * overlap_h))
    return overlaps

print("=" * 60)
print("BEFORE optimization")
print("=" * 60)
for i, box in enumerate(boxes):
    w = box[2] - box[0]
    h = box[3] - box[1]
    print(f"  Room {i}: x={box[0]:3d} y={box[1]:3d}  w={w:3d} h={h:3d}  area={w*h}")

overlaps_before = check_overlaps(boxes)
print(f"\n  Overlapping pairs: {len(overlaps_before)}")
for i, j, area in overlaps_before:
    print(f"    Room {i} x Room {j}  overlap area = {area} px²")

# ---------------------------------------------------------------------------
# Run optimizer
# ---------------------------------------------------------------------------
print("\nRunning CP-SAT optimizer...")
result, status = optimize_layout(boxes, types, edges, boundary=boundary, timeout=10.0)
print(f"Status: {status}")

# ---------------------------------------------------------------------------
# Print result
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("AFTER optimization")
print("=" * 60)
for i, box in enumerate(result):
    w = box[2] - box[0]
    h = box[3] - box[1]
    orig = boxes[i]
    dx = abs(int(box[0]) - int(orig[0]))
    dy = abs(int(box[1]) - int(orig[1]))
    print(f"  Room {i}: x={box[0]:3d} y={box[1]:3d}  w={w:3d} h={h:3d}  area={w*h}  "
          f"(moved dx={dx} dy={dy})")

overlaps_after = check_overlaps(result)
print(f"\n  Overlapping pairs: {len(overlaps_after)}")
if overlaps_after:
    for i, j, area in overlaps_after:
        print(f"    Room {i} x Room {j}  overlap area = {area} px²")
else:
    print("  No overlaps — constraint satisfied.")

# Boundary check
print("\n  Boundary check (all rooms inside 0..240):")
all_inside = True
for i, box in enumerate(result):
    inside = box[0] >= 0 and box[1] >= 0 and box[2] <= 240 and box[3] <= 240
    if not inside:
        print(f"    Room {i} OUTSIDE boundary: {box}")
        all_inside = False
if all_inside:
    print("  All rooms inside boundary — constraint satisfied.")

# ---------------------------------------------------------------------------
# Visualise before / after side by side
# ---------------------------------------------------------------------------
ROOM_NAMES  = ['LivingRoom', 'MasterRoom', 'Kitchen', 'Bathroom', 'DiningRoom']
ROOM_COLORS = [
    '#e6194b',  # LivingRoom  red
    '#3cb44b',  # MasterRoom  green
    '#aaffc3',  # Kitchen     mint
    '#0082c8',  # Bathroom    blue
    '#f58230',  # DiningRoom  orange
]

try:
    import matplotlib
    matplotlib.use('TkAgg')   # works on Windows; fall back below if it fails
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle('CP-SAT Optimizer — Before vs After', fontsize=13)

    for ax, data, title in [
        (axes[0], boxes,  f'Before  ({len(overlaps_before)} overlaps)'),
        (axes[1], result, f'After   ({len(overlaps_after)} overlaps)'),
    ]:
        ax.set_title(title)
        ax.set_xlim(0, 240)
        ax.set_ylim(240, 0)   # y=0 at top (image convention)
        ax.set_aspect('equal')
        ax.set_xlabel('x (px)')
        ax.set_ylabel('y (px)')

        # Draw boundary
        ax.add_patch(patches.Rectangle(
            (0, 0), 240, 240,
            linewidth=2, edgecolor='black', facecolor='#f5f5f5'
        ))

        for i, box in enumerate(data):
            x0, y0, x1, y1 = box
            w, h = x1 - x0, y1 - y0
            color = ROOM_COLORS[i % len(ROOM_COLORS)]
            rect = patches.Rectangle(
                (x0, y0), w, h,
                linewidth=1.5, edgecolor='black',
                facecolor=color, alpha=0.6
            )
            ax.add_patch(rect)
            ax.text(
                x0 + w / 2, y0 + h / 2,
                f"{ROOM_NAMES[i]}\n{w}×{h}",
                ha='center', va='center', fontsize=7, fontweight='bold'
            )

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), 'optimizer_test_result.png')
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved to: {out_path}")
    plt.show()

except Exception as e:
    print(f"\n(Visualisation skipped: {e})")
