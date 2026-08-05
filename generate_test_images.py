#!/usr/bin/env python3
"""
Generate synthetic test images for the calibration frame measurement system.

Renders the default frame (170x170mm outer, 10mm border, 1mm fillets,
150x150mm inner window) at a known scale with objects of exactly known
dimensions, then applies camera-like effects (perspective, lighting
gradient, blur, noise). Ground-truth sizes are printed for verification.
"""

import cv2
import numpy as np

PX_PER_MM = 10.0
OUTER_MM = 170.0
BORDER_MM = 10.0
FILLET_MM = 1.0


def rounded_rect(img, x0, y0, x1, y1, r, color):
    """Draw a filled rectangle with rounded corners (radius r px)"""
    r = int(r)
    cv2.rectangle(img, (x0 + r, y0), (x1 - r, y1), color, -1)
    cv2.rectangle(img, (x0, y0 + r), (x1, y1 - r), color, -1)
    for cx, cy in [(x0 + r, y0 + r), (x1 - r, y0 + r), (x1 - r, y1 - r), (x0 + r, y1 - r)]:
        cv2.circle(img, (cx, cy), r, color, -1)


def base_scene(margin_mm=25.0, table_gray=185):
    """Blank scene: gray table, black frame, white window. Returns (img, origin_px)
    where origin_px is the pixel position of the window's top-left corner."""
    outer = int(OUTER_MM * PX_PER_MM)
    margin = int(margin_mm * PX_PER_MM)
    size = outer + 2 * margin
    img = np.full((size, size, 3), table_gray, dtype=np.uint8)

    fillet = FILLET_MM * PX_PER_MM
    border = int(BORDER_MM * PX_PER_MM)
    # black frame (outer edge), then white window (inner edge)
    rounded_rect(img, margin, margin, margin + outer, margin + outer, fillet, (10, 10, 10))
    rounded_rect(img, margin + border, margin + border,
                 margin + outer - border, margin + outer - border, fillet, (255, 255, 255))
    origin = (margin + border, margin + border)
    return img, origin


def add_rect_object(img, origin, x_mm, y_mm, w_mm, h_mm, color=(60, 55, 50)):
    ox, oy = origin
    x0 = int(ox + x_mm * PX_PER_MM)
    y0 = int(oy + y_mm * PX_PER_MM)
    x1 = int(x0 + w_mm * PX_PER_MM)
    y1 = int(y0 + h_mm * PX_PER_MM)
    cv2.rectangle(img, (x0, y0), (x1, y1), color, -1)


def add_disc_object(img, origin, cx_mm, cy_mm, d_mm, color=(45, 60, 70)):
    ox, oy = origin
    cv2.circle(img, (int(ox + cx_mm * PX_PER_MM), int(oy + cy_mm * PX_PER_MM)),
               int(d_mm / 2 * PX_PER_MM), color, -1)


def camera_effects(img, perspective=None, blur=1.5, noise=4, vignette=0.15, seed=0):
    """Apply perspective warp, lighting gradient, blur and sensor noise"""
    h, w = img.shape[:2]
    out = img.astype(np.float32)

    # lighting gradient (brighter top-left, darker bottom-right)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    grad = 1.0 + vignette * (1.0 - (xx / w + yy / h))
    out *= grad[..., None]

    if perspective is not None:
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        M = cv2.getPerspectiveTransform(src, np.float32(perspective))
        out = cv2.warpPerspective(out, M, (w, h), borderValue=(185, 185, 185))

    if blur > 0:
        out = cv2.GaussianBlur(out, (0, 0), blur)

    rng = np.random.default_rng(seed)
    out += rng.normal(0, noise, out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    tests = []

    # 1. Straight-on: 60x25mm rectangle
    img, o = base_scene()
    add_rect_object(img, o, 45, 60, 60, 25)
    tests.append(("test_rect_flat.png", camera_effects(img, seed=1),
                  "rectangle 60.0 x 25.0 mm"))

    # 2. Straight-on: 40mm diameter disc
    img, o = base_scene()
    add_disc_object(img, o, 75, 75, 40)
    tests.append(("test_disc_flat.png", camera_effects(img, seed=2),
                  "disc 40.0 mm diameter, area 1256.6 mm2"))

    # 3. Camera at an angle: same 60x25mm rectangle
    img, o = base_scene()
    add_rect_object(img, o, 45, 60, 60, 25)
    h, w = img.shape[:2]
    persp = [[w * 0.08, h * 0.06], [w * 0.94, h * 0.02],
             [w * 0.99, h * 0.97], [w * 0.03, h * 0.92]]
    tests.append(("test_rect_angle.png", camera_effects(img, perspective=persp, seed=3),
                  "rectangle 60.0 x 25.0 mm (perspective)"))

    # 4. Two shapes, harder lighting: L-shape (two rects) 70x50mm bbox
    img, o = base_scene()
    add_rect_object(img, o, 40, 50, 70, 15)
    add_rect_object(img, o, 40, 50, 15, 50)
    tests.append(("test_lshape_flat.png", camera_effects(img, vignette=0.25, noise=6, seed=4),
                  "L-shape bbox 70.0 x 50.0 mm"))

    for name, im, truth in tests:
        cv2.imwrite(name, im)
        print(f"{name}: ground truth = {truth}")


if __name__ == "__main__":
    main()
