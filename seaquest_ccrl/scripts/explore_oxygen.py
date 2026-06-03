"""Empirically determine the oxygen-bar mask rectangle and confirm object categories.

Dives the submarine to deplete oxygen, recording OxygenBar / OxygenBarDepleted
bounding boxes across the full->empty range. Prints the UNION bbox (+ label strip)
to hardcode as OXY_MASK_RECT. Per skill Invariant 3: mask the FULL strip, never the
filled width.
"""
import numpy as np
from collections import Counter
from ocatari.core import OCAtari

DOWN = 5  # dive to stay underwater so oxygen depletes

env = OCAtari("ALE/Seaquest-v5", mode="ram", hud=True, render_mode="rgb_array",
              frameskip=4, repeat_action_probability=0.0)
env.reset(seed=0)

cats_seen = Counter()
oxy_boxes = []      # (x,y,w,h) of OxygenBar (filled)
dep_boxes = []      # (x,y,w,h) of OxygenBarDepleted (empty)
oxy_widths = []

def boxes_by_cat(objs, cat):
    return [(o.x, o.y, o.w, o.h) for o in objs if o.category == cat]

for step in range(1500):
    _, _, term, trunc, _ = env.step(DOWN)
    objs = [o for o in env.objects if o.category != "NoObject"]
    for o in objs:
        cats_seen[o.category] += 1
    ob = boxes_by_cat(objs, "OxygenBar")
    dp = boxes_by_cat(objs, "OxygenBarDepleted")
    oxy_boxes += ob
    dep_boxes += dp
    if ob:
        oxy_widths.append(ob[0][2])
    if term or trunc:
        env.reset()

frame = env.render()
print("=== frame shape:", frame.shape, frame.dtype)
print("\n=== object categories seen (count over run) ===")
for c, n in cats_seen.most_common():
    print(f"  {c:22s} {n}")

def union(boxes):
    if not boxes:
        return None
    xs0 = min(b[0] for b in boxes); ys0 = min(b[1] for b in boxes)
    xs1 = max(b[0] + b[2] for b in boxes); ys1 = max(b[1] + b[3] for b in boxes)
    return (xs0, ys0, xs1 - xs0, ys1 - ys0)

print("\n=== OxygenBar (filled) observed ===")
print("  n samples:", len(oxy_boxes), "| width range:",
      (min(oxy_widths) if oxy_widths else None, max(oxy_widths) if oxy_widths else None))
print("  union bbox (x,y,w,h):", union(oxy_boxes))
print("=== OxygenBarDepleted (empty) observed ===")
print("  n samples:", len(dep_boxes), "| union bbox (x,y,w,h):", union(dep_boxes))

full_union = union(oxy_boxes + dep_boxes)
print("\n=== UNION OxygenBar ∪ OxygenBarDepleted (x,y,w,h):", full_union)
if full_union:
    x, y, w, h = full_union
    # pad to cover the OXYGEN label strip + antialiasing; clamp to frame
    PAD_X, PAD_TOP, PAD_BOT = 3, 8, 3
    rx = max(0, x - PAD_X)
    ry = max(0, y - PAD_TOP)
    rw = min(160 - rx, w + 2 * PAD_X)
    rh = min(210 - ry, h + PAD_TOP + PAD_BOT)
    print(f"=== SUGGESTED OXY_MASK_RECT (x,y,w,h) = ({rx}, {ry}, {rw}, {rh})")
    print(f"    i.e. rows [{ry}:{ry+rh}], cols [{rx}:{rx+rw}]")
env.close()
