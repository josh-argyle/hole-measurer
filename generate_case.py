#!/usr/bin/env python3
"""
Case / organizer generator.

Takes measured outlines (warped px, 10 px/mm) and produces a printable
STL: a block or Gridfinity-sized insert with each object recessed as a
pocket, or a form-fitting shell. Can also emit the clearance-offset
outline as a DXF for CAD work.

Usage: generate_case.py <input.json> <output-file>

Input JSON:
{
  "objects":      [[[x, y], ...], ...],   # outlines in warped px, y-down
  "px_per_mm":    10,
  "mode":         "block" | "gridfinity" | "shell" | "offset-dxf",
  "clearance_mm": 0.3,    # gap around the object so it drops in
  "depth_mm":     15,     # pocket depth (or wall height for shell)
  "floor_mm":     2,      # material left under the pocket
  "margin_mm":    5,      # block border around the outlines
  "wall_mm":      1.6,    # shell / pocket wall thickness
  "hollow":       false,  # open the underside, honeycomb ribs support decks
  "lip_mm":       0,      # roundover radius on the pocket opening
  "fillet_mm":    0       # fillet radius where pocket wall meets floor
}

Prints a single JSON line: {"ok": true, "out": "<path>", ...} or an error.
"""

import json
import struct
import sys

import numpy as np
from manifold3d import CrossSection, FillRule, JoinType, Manifold, OpType

# Gridfinity spec: 42 mm grid, 7 mm height unit, 0.5 mm bin clearance.
# Base profile per cell, bottom up: 0.8 mm 45-degree chamfer, 1.8 mm
# straight, 2.15 mm 45-degree chamfer. Corner radii follow the chamfers
# (0.8 -> 1.6 -> 3.75) because each step is a round outward offset.
GRID = 42.0
HEIGHT_UNIT = 7.0
BIN_CLEARANCE = 0.5
BASE_CHAMFER1 = 0.8
BASE_STRAIGHT = 1.8
BASE_CHAMFER2 = 2.15
BASE_H = BASE_CHAMFER1 + BASE_STRAIGHT + BASE_CHAMFER2  # 4.75
CELL_TOP = GRID - BIN_CLEARANCE                          # 41.5
CELL_BOTTOM = CELL_TOP - 2 * (BASE_CHAMFER1 + BASE_CHAMFER2)  # 35.6
CORNER_R = 3.75


def rounded_rect(w, h, r):
    """Centered rounded rectangle as a CrossSection"""
    r = min(r, w / 2 - 0.01, h / 2 - 0.01)
    if r <= 0:
        return CrossSection.square((w, h), True)
    return CrossSection.square((w - 2 * r, h - 2 * r), True).offset(
        r, JoinType.Round, circular_segments=32)


def slab(cs, z0, z1):
    """Extrude a CrossSection between two heights"""
    return Manifold.extrude(cs, z1 - z0).translate((0, 0, z0))


def chamfer(cs_bottom, cs_top, z0, z1):
    """Loft two convex sections via hull (both are rounded squares)"""
    eps = 0.01
    return Manifold.batch_hull([slab(cs_bottom, z0, z0 + eps),
                                slab(cs_top, z1 - eps, z1)])


def gridfinity_base(nx, ny):
    """The stack of per-cell base pads that drops into a baseplate"""
    bottom = rounded_rect(CELL_BOTTOM, CELL_BOTTOM, CORNER_R - BASE_CHAMFER1 - BASE_CHAMFER2)
    mid = rounded_rect(CELL_BOTTOM + 2 * BASE_CHAMFER1,
                       CELL_BOTTOM + 2 * BASE_CHAMFER1,
                       CORNER_R - BASE_CHAMFER2)
    top = rounded_rect(CELL_TOP, CELL_TOP, CORNER_R)

    cell = (chamfer(bottom, mid, 0, BASE_CHAMFER1)
            + slab(mid, BASE_CHAMFER1, BASE_CHAMFER1 + BASE_STRAIGHT)
            + chamfer(mid, top, BASE_CHAMFER1 + BASE_STRAIGHT, BASE_H))

    pads = []
    x0 = -(nx - 1) * GRID / 2
    y0 = -(ny - 1) * GRID / 2
    for i in range(nx):
        for j in range(ny):
            pads.append(cell.translate((x0 + i * GRID, y0 + j * GRID, 0)))
    return sum(pads[1:], pads[0])


def hex_lattice(w, h, pitch=12.0, rib=1.0):
    """
    Honeycomb rib pattern covering a centered w x h area. Hollowed cavities
    keep these ribs so the deck above prints as short bridges instead of
    one long unsupported span (each bridge <= pitch).
    """
    R = pitch / np.sqrt(3)  # circumradius: across-flats equals pitch
    ang = np.arange(6) * np.pi / 3 + np.pi / 6  # pointy-top

    def hexagon(r):
        return CrossSection([[(r * np.cos(a), r * np.sin(a)) for a in ang]])

    ring = hexagon(R + rib / 2) - hexagon(R - rib / 2)
    dy = 1.5 * R
    nx = int(np.ceil(w / pitch / 2)) + 1
    ny = int(np.ceil(h / dy / 2)) + 1
    cells = []
    for j in range(-ny, ny + 1):
        xoff = pitch / 2 if j % 2 else 0
        for i in range(-nx, nx + 1):
            cells.append(ring.translate((i * pitch + xoff, j * dy)))
    return sum(cells[1:], cells[0])


def _sweep_profile(cs, profile_2d):
    """
    Sweep a convex 2D profile (in the (outward, z) plane) along every
    boundary of a CrossSection: one convex hull per boundary edge plus one
    per corner. Gives a genuinely smooth swept surface - no stacked steps.

    profile_2d: list of (s, z) points; s > 0 is outward from the boundary.
    Returns a single Manifold, or None if the boundary was degenerate.
    """
    prof = np.asarray(profile_2d, dtype=np.float64)
    solids = []
    for poly in cs.simplify(0.02).to_polygons():
        P = np.asarray(poly, dtype=np.float64)
        keep = np.linalg.norm(P - np.roll(P, 1, axis=0), axis=1) > 1e-6
        P = P[keep]
        n = len(P)
        if n < 3:
            continue

        def disk(p, m):
            """Profile polygon placed at boundary point p, outward normal m"""
            return np.column_stack([p[0] + m[0] * prof[:, 0],
                                    p[1] + m[1] * prof[:, 0],
                                    prof[:, 1]])

        normals = []
        for i in range(n):
            e = P[(i + 1) % n] - P[i]
            length = np.linalg.norm(e)
            # Positive-area loops are CCW, so (ey, -ex) points outward
            normals.append(np.array([e[1], -e[0]]) / length if length > 1e-9
                           else None)
        for i in range(n):
            if normals[i] is None:
                continue
            j = (i + 1) % n
            a = disk(P[i], normals[i])
            b = disk(P[j], normals[i])
            solids.append(Manifold.hull_points(np.vstack([a, b]).tolist()))
            if normals[j] is not None:
                c = disk(P[j], normals[j])
                solids.append(Manifold.hull_points(np.vstack([b, c]).tolist()))
    if not solids:
        return None
    return Manifold.batch_boolean(solids, OpType.Add)


def lip_cutter(cs, z_top, r):
    """
    Roundover cutter for a pocket opening: a quarter-round lead-in swept
    along the boundary, tangent to the wall and the top face.
    """
    arc = 14
    prof = [(0.0, z_top + 1), (r, z_top + 1)]
    for k in range(arc + 1):
        d = r * k / arc
        s = r - np.sqrt(max(r * r - (r - d) ** 2, 0.0))
        prof.append((s, z_top - d))
    return _sweep_profile(cs, prof)


def floor_fillet(cs, z_floor, r):
    """
    Concave fillet where the pocket wall meets the pocket floor: a
    quarter-round swept along the boundary, tangent to wall and floor.
    Returns a Manifold or None.
    """
    arc = 14
    prof = [(0.0, z_floor)]
    for k in range(arc + 1):
        h = r * k / arc
        a = r - np.sqrt(max(r * r - (r - h) ** 2, 0.0))
        prof.append((-a, z_floor + h))
    return _sweep_profile(cs, prof)


def outlines_to_cross_section(objects, px_per_mm, clearance):
    """Convert px outlines to a single mm CrossSection, offset by the
    clearance. Y is flipped so the pocket matches the photo seen from
    above (STL is Y-up, images are Y-down)."""
    polys = []
    for pts in objects:
        arr = np.asarray(pts, dtype=np.float64)
        if len(arr) < 3:
            continue
        arr = arr / px_per_mm
        arr[:, 1] = -arr[:, 1]
        polys.append(arr.tolist())
    if not polys:
        raise ValueError("No usable outlines")
    cs = CrossSection(polys, FillRule.EvenOdd)
    if clearance > 0:
        cs = cs.offset(clearance, JoinType.Round, circular_segments=32)
    if cs.area() <= 0:
        raise ValueError("Outline collapsed - check the path isn't self-crossing")
    return cs


def write_stl(manifold, path):
    mesh = manifold.to_mesh()
    verts = np.asarray(mesh.vert_properties, dtype=np.float32)[:, :3]
    tris = np.asarray(mesh.tri_verts, dtype=np.int64)
    v0, v1, v2 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    lens = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(lens, 1e-12)

    with open(path, 'wb') as f:
        f.write(b'Measur case generator'.ljust(80, b'\0'))
        f.write(struct.pack('<I', len(tris)))
        block = np.zeros((len(tris), 50), dtype=np.uint8)
        data = np.hstack([normals, v0, v1, v2]).astype('<f4')
        block[:, :48] = data.view(np.uint8).reshape(len(tris), 48)
        f.write(block.tobytes())


def write_offset_dxf(cs, path):
    lines = ['0', 'SECTION', '2', 'ENTITIES']
    for poly in cs.to_polygons():
        lines += ['0', 'POLYLINE', '8', '0', '66', '1', '70', '1']
        for x, y in poly:
            lines += ['0', 'VERTEX', '8', '0', '10', f'{x:.3f}', '20', f'{y:.3f}']
        lines += ['0', 'SEQEND']
    lines += ['0', 'ENDSEC', '0', 'EOF']
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    spec = json.load(open(sys.argv[1]))
    out_path = sys.argv[2]

    px_per_mm = float(spec.get('px_per_mm', 10))
    mode = spec.get('mode', 'block')
    clearance = float(spec.get('clearance_mm', 0.3))
    depth = float(spec.get('depth_mm', 15))
    floor = float(spec.get('floor_mm', 2))
    margin = float(spec.get('margin_mm', 5))
    wall = float(spec.get('wall_mm', 1.6))
    hollow = bool(spec.get('hollow', False))
    lip = max(0.0, float(spec.get('lip_mm', 0)))
    fillet = max(0.0, float(spec.get('fillet_mm', 0)))

    pockets = outlines_to_cross_section(spec['objects'], px_per_mm, clearance)
    x0, y0, x1, y1 = pockets.bounds()
    # Center the content on the origin
    pockets = pockets.translate((-(x0 + x1) / 2, -(y0 + y1) / 2))
    content_w, content_h = x1 - x0, y1 - y0

    info = {'ok': True, 'out': out_path, 'mode': mode}

    if mode == 'offset-dxf':
        write_offset_dxf(pockets, out_path)
        print(json.dumps(info))
        return

    if mode == 'shell':
        outer = pockets.offset(wall, JoinType.Round, circular_segments=32)
        base = slab(outer, 0, floor)
        walls = slab(outer - pockets, floor, floor + depth)
        solid = base + walls
        pocket_top, pocket_floor = floor + depth, floor
        info['size_mm'] = [round(content_w + 2 * wall, 1),
                          round(content_h + 2 * wall, 1),
                          round(floor + depth, 1)]

    elif mode == 'gridfinity':
        nx = max(1, int(np.ceil((content_w + 2 * margin) / GRID)))
        ny = max(1, int(np.ceil((content_h + 2 * margin) / GRID)))
        total_h = max(HEIGHT_UNIT,
                      HEIGHT_UNIT * np.ceil((floor + depth) / HEIGHT_UNIT))
        body_w = nx * GRID - BIN_CLEARANCE
        body_h = ny * GRID - BIN_CLEARANCE
        if content_w > body_w - 2 or content_h > body_h - 2:
            print(json.dumps({'ok': False,
                              'error': 'Objects larger than the gridfinity block'}))
            return
        footprint = rounded_rect(body_w, body_h, CORNER_R)
        body = slab(footprint, BASE_H, total_h)
        solid = gridfinity_base(nx, ny) + body
        # Pocket cut straight down from the top face
        cut = slab(pockets, total_h - depth, total_h + 1)
        solid = solid - cut
        pocket_top, pocket_floor = total_h, total_h - depth
        if hollow:
            guard = pockets.offset(wall, JoinType.Round, circular_segments=32)
            lattice = hex_lattice(body_w, body_h)
            # Around the pockets: hollow up to a deck under the top face
            body_inner = footprint.offset(-wall, JoinType.Round) - guard - lattice
            if body_inner.area() > 0 and total_h - floor > BASE_H:
                solid = solid - slab(body_inner, BASE_H, total_h - floor)
            # Under the pockets: keep a floor-thick deck as the pocket floor
            under = pockets.offset(-wall, JoinType.Round) - lattice
            if under.area() > 0 and total_h - depth - floor > BASE_H + 0.4:
                solid = solid - slab(under, BASE_H, total_h - depth - floor)
            # Base pads: hollow their interiors, keep the engagement walls
            pad_size = CELL_BOTTOM - 2 * wall
            if pad_size > 2:
                pad_inner = rounded_rect(pad_size, pad_size, 0.8)
                pads = []
                x0 = -(nx - 1) * GRID / 2
                y0 = -(ny - 1) * GRID / 2
                for i in range(nx):
                    for j in range(ny):
                        pads.append(pad_inner.translate((x0 + i * GRID,
                                                         y0 + j * GRID)))
                pads_cs = sum(pads[1:], pads[0]) - guard - lattice
                if pads_cs.area() > 0:
                    solid = solid - slab(pads_cs, -1, BASE_H + 0.01)
        info['grid'] = [nx, ny]
        info['size_mm'] = [round(body_w, 1), round(body_h, 1), round(total_h, 1)]

    else:  # plain block
        block_w = content_w + 2 * margin
        block_h = content_h + 2 * margin
        total_h = floor + depth
        footprint = rounded_rect(block_w, block_h, min(3.0, margin))
        block = slab(footprint, 0, total_h)
        cut = slab(pockets, floor, total_h + 1)
        solid = block - cut
        pocket_top, pocket_floor = total_h, floor
        if hollow:
            guard = pockets.offset(wall, JoinType.Round, circular_segments=32)
            lattice = hex_lattice(block_w, block_h)
            # Around the pockets: hollow up to a deck under the top face,
            # keeping honeycomb ribs so the deck prints as short bridges.
            # Under the pockets nothing needs removing in block mode: the
            # pocket already reaches down to a floor-thick deck.
            cavity = footprint.offset(-wall, JoinType.Round) - guard - lattice
            if cavity.area() > 0 and total_h - floor > 0.4:
                solid = solid - slab(cavity, -1, total_h - floor)
        info['size_mm'] = [round(block_w, 1), round(block_h, 1), round(total_h, 1)]

    # Lip roundover around the pocket opening (eases dropping the tool in)
    # and a concave fillet where the pocket wall meets the pocket floor.
    lip_eff = min(lip, depth / 2)
    if hollow:
        # Don't break through the deck or the pocket walls into the cavity
        lip_eff = min(lip_eff, wall, max(0.0, floor - 0.2))
    if lip_eff > 0.1:
        cutter = lip_cutter(pockets, pocket_top, lip_eff)
        if cutter is not None:
            solid = solid - cutter
            info['lip_mm'] = round(lip_eff, 2)
    fillet_eff = min(fillet, depth / 2)
    if fillet_eff > 0.1:
        ring = floor_fillet(pockets, pocket_floor, fillet_eff)
        if ring is not None:
            solid = solid + ring
            info['fillet_mm'] = round(fillet_eff, 2)

    if solid.volume() <= 0:
        print(json.dumps({'ok': False, 'error': 'Generated an empty solid - '
                          'check depth/floor values'}))
        return

    write_stl(solid, out_path)
    info['triangles'] = int(np.asarray(solid.to_mesh().tri_verts).shape[0])
    # Printed weight at 100% of this geometry (PLA, 1.24 g/cm3); slicer
    # infill settings only matter for whatever solid volume remains
    info['grams_pla'] = round(solid.volume() / 1000.0 * 1.24, 1)
    print(json.dumps(info))


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(json.dumps({'ok': False, 'error': str(e)}))
        sys.exit(1)
