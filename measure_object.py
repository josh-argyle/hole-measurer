#!/usr/bin/env python3
"""
Calibration Frame Object Measurement System
Measures 2D objects placed inside a black rectangular frame.

Calibration works by detecting the inner edge of the black frame (the
white window the object sits in) and computing a homography from its
4 corners. Only what is inside the window is measured.
"""

import cv2
import numpy as np
import argparse
from pathlib import Path

# Annotated results are saved here by default (created on demand)
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


# Inner-window dimensions of the calibration frame (mm).
# Default frame: 170x170mm outer, 10mm border -> 150x150mm inner opening.
# Measure the INNER opening of your printed frame and pass --width/--height
# if it differs.
DEFAULT_INNER_WIDTH_MM = 150.0
DEFAULT_INNER_HEIGHT_MM = 150.0


class CalibrationFrame:
    """Detects the black frame's inner window and calibrates the view"""

    def __init__(self, inner_width_mm=DEFAULT_INNER_WIDTH_MM,
                 inner_height_mm=DEFAULT_INNER_HEIGHT_MM):
        """
        Args:
            inner_width_mm: Width of the frame's inner opening (mm)
            inner_height_mm: Height of the frame's inner opening (mm)
        """
        self.inner_width_mm = inner_width_mm
        self.inner_height_mm = inner_height_mm
        self.homography_matrix = None
        self.mm_per_pixel = None

    @staticmethod
    def _order_corners(pts):
        """Order 4 points as top-left, top-right, bottom-right, bottom-left"""
        pts = pts.reshape(4, 2).astype(np.float32)
        s = pts.sum(axis=1)
        d = pts[:, 0] - pts[:, 1]
        return np.array([
            pts[np.argmin(s)],   # top-left
            pts[np.argmax(d)],   # top-right
            pts[np.argmax(s)],   # bottom-right
            pts[np.argmin(d)],   # bottom-left
        ], dtype=np.float32)

    @staticmethod
    def _corners_from_edge_lines(contour, rough_corners):
        """
        Refine corners by fitting a straight line to each of the 4 edges and
        intersecting adjacent lines. Immune to rounded (filleted) corners,
        since points near the corners are excluded from the fits.

        Args:
            contour: The window contour from findContours
            rough_corners: (4, 2) rough corners ordered TL, TR, BR, BL

        Returns:
            (4, 2) float32 refined corners, same order. Falls back to
            rough_corners if any edge has too few points to fit.
        """
        pts = contour.reshape(-1, 2).astype(np.float32)
        lines = []

        for i in range(4):
            c1 = rough_corners[i]
            c2 = rough_corners[(i + 1) % 4]
            edge_vec = c2 - c1
            edge_len = np.linalg.norm(edge_vec)
            if edge_len < 1:
                return rough_corners
            direction = edge_vec / edge_len

            # Position of each contour point along the edge (0..1) and its
            # perpendicular distance from the edge line
            rel = pts - c1
            t = rel @ direction / edge_len
            perp = np.abs(rel[:, 0] * direction[1] - rel[:, 1] * direction[0])

            # Keep points on the middle 70% of the edge (skips fillets) that
            # lie close to it (skips the other edges)
            mask = (t > 0.15) & (t < 0.85) & (perp < 0.05 * edge_len)
            edge_pts = pts[mask]
            if len(edge_pts) < 10:
                return rough_corners

            vx, vy, x0, y0 = cv2.fitLine(edge_pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
            lines.append((x0, y0, vx, vy))

        # Intersect adjacent edge lines: corner i = line[i-1] x line[i]
        corners = np.zeros((4, 2), dtype=np.float32)
        for i in range(4):
            x1, y1, vx1, vy1 = lines[i - 1]
            x2, y2, vx2, vy2 = lines[i]
            denom = vx1 * vy2 - vy1 * vx2
            if abs(denom) < 1e-9:
                return rough_corners
            s = ((x2 - x1) * vy2 - (y2 - y1) * vx2) / denom
            corners[i] = (x1 + s * vx1, y1 + s * vy1)

        return corners

    def detect_inner_window(self, image, debug=False):
        """
        Find the inner edge of the black frame.

        Returns:
            (4, 2) float32 array of corners ordered TL, TR, BR, BL
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, white = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # RETR_CCOMP: bright-region outer boundaries have no parent, even when
        # nested inside another bright region's hole (e.g. window inside frame
        # inside a bright table surface). RETR_EXTERNAL would drop those.
        contours, hierarchy = cv2.findContours(white, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("No bright regions found - is the frame in the image?")

        h, w = gray.shape
        image_area = h * w

        # The window is a large bright region fully enclosed by the black
        # frame, so it cannot touch the image border. Bright regions outside
        # the frame (e.g. a white table) do touch the border - skip them.
        def touches_border(c):
            x, y, cw, ch = cv2.boundingRect(c)
            return x <= 1 or y <= 1 or x + cw >= w - 1 or y + ch >= h - 1

        candidates = [c for c, hier in zip(contours, hierarchy[0])
                      if hier[3] == -1 and not touches_border(c)]
        if not candidates:
            raise ValueError("Could not find the frame window - the black frame "
                             "must fully surround a bright area and be fully in view")

        window = max(candidates, key=cv2.contourArea)
        if cv2.contourArea(window) < 0.1 * image_area:
            raise ValueError("Frame window too small - check that the black frame "
                             "fills a good portion of the photo")

        peri = cv2.arcLength(window, True)
        approx = cv2.approxPolyDP(window, 0.02 * peri, True)
        if len(approx) != 4:
            # Perspective still yields a quadrilateral; fall back to the
            # minimum-area rectangle if the contour was noisy
            approx = cv2.boxPoints(cv2.minAreaRect(window))

        rough = self._order_corners(np.array(approx, dtype=np.float32))
        corners = self._corners_from_edge_lines(window, rough)

        if debug:
            print(f"Inner window corners (TL, TR, BR, BL): {corners.tolist()}")
            dbg = image.copy()
            cv2.polylines(dbg, [corners.astype(np.int32)], True, (0, 255, 0), 2)
            for (x, y) in corners:
                cv2.circle(dbg, (int(x), int(y)), 6, (0, 0, 255), -1)
            cv2.imshow('Detected Inner Window', dbg)

        return corners

    def calibrate(self, image, debug=False):
        """
        Detect the frame window and compute the pixel->mm homography.

        Returns:
            True if calibration succeeded, False otherwise
        """
        try:
            corners = self.detect_inner_window(image, debug=debug)
        except ValueError as e:
            print(f"Error: {e}")
            return False

        known = np.array([
            [0, 0],
            [self.inner_width_mm, 0],
            [self.inner_width_mm, self.inner_height_mm],
            [0, self.inner_height_mm],
        ], dtype=np.float32)

        self.homography_matrix = cv2.getPerspectiveTransform(corners, known)

        h00, h01 = self.homography_matrix[0, 0], self.homography_matrix[0, 1]
        h10, h11 = self.homography_matrix[1, 0], self.homography_matrix[1, 1]
        scale_x = np.sqrt(h00**2 + h10**2)
        scale_y = np.sqrt(h01**2 + h11**2)
        self.mm_per_pixel = (scale_x + scale_y) / 2.0

        print(f"Calibration successful! Scale: {self.mm_per_pixel:.4f} mm/pixel")
        return True

    def warp_to_calibrated_view(self, image):
        """
        Warp the inner window to a top-down view at 10 pixels per mm.
        Everything outside the window is cropped away.
        """
        if self.homography_matrix is None:
            raise ValueError("Must calibrate before warping")

        scale = 10  # pixels per mm
        output_width = int(round(self.inner_width_mm * scale))
        output_height = int(round(self.inner_height_mm * scale))

        scale_matrix = np.array([
            [scale, 0, 0],
            [0, scale, 0],
            [0, 0, 1]
        ], dtype=np.float64)

        adjusted_homography = scale_matrix @ self.homography_matrix
        return cv2.warpPerspective(image, adjusted_homography,
                                   (output_width, output_height))


class ObjectMeasurer:
    """Measures objects in the calibrated frame"""

    def __init__(self, calibration_frame):
        self.frame = calibration_frame

    def segment_object(self, warped_image, threshold=200, debug=False):
        """
        Segment object from white background

        Returns:
            Binary mask of the object
        """
        gray = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)

        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        if debug:
            cv2.imshow('Object Segmentation', binary)

        return binary

    def _score_contour(self, contour, gradient_mag, edges, image_shape):
        """
        Score a contour based on multiple quality metrics

        Args:
            gradient_mag: Precomputed gradient magnitude image (Sobel)
            edges: Precomputed Canny edge map (frame border masked off)

        Returns a score between 0 and 1 (higher = better object candidate)
        """
        h, w = image_shape[:2]
        total_area = h * w

        area = cv2.contourArea(contour)
        x, y, cw, ch = cv2.boundingRect(contour)
        bbox_area = cw * ch

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)

        # 1. SIZE SCORE - Prefer objects that are 5-40% of frame area
        optimal_area_ratio = 0.15
        area_ratio = area / total_area
        size_score = np.exp(-((area_ratio - optimal_area_ratio)**2) / (2 * 0.1**2))

        # 2. SHAPE SCORE - Prefer solid, compact shapes
        solidity = area / hull_area if hull_area > 0 else 0
        extent = area / bbox_area if bbox_area > 0 else 0
        shape_score = (solidity * 0.6 + extent * 0.4)

        # 3. COMPLETENESS SCORE - Prefer shapes that fill their bounding box well
        aspect_ratio = max(cw, ch) / max(min(cw, ch), 1)
        aspect_penalty = 1.0 if aspect_ratio < 15 else (15.0 / aspect_ratio)
        completeness_score = extent * aspect_penalty

        # 4. SHARPNESS SCORE - Real object edges are sharp, shadows are soft.
        # Use the 25th percentile of gradient magnitude along the boundary:
        # a contour that includes any soft (shadow) stretch scores low even
        # if the rest of its outline is crisp.
        boundary = contour.reshape(-1, 2)
        step = max(1, len(boundary) // 200)
        samples = boundary[::step]
        ys = np.clip(samples[:, 1], 0, h - 1)
        xs = np.clip(samples[:, 0], 0, w - 1)
        boundary_grad = float(np.percentile(gradient_mag[ys, xs], 25))
        sharpness_score = min(1.0, boundary_grad / 40.0)

        # 5. EDGE COVERAGE - Fraction of the image's crisp edges that lie
        # inside this candidate. The whole object contains (nearly) all of
        # them; a fragment of it contains few.
        total_edges = np.count_nonzero(edges)
        if total_edges > 0:
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            mask = cv2.dilate(mask, np.ones((7, 7), np.uint8))
            coverage_score = np.count_nonzero(edges[mask > 0]) / total_edges
        else:
            coverage_score = 0.5

        total_score = (
            0.10 * size_score +
            0.10 * shape_score +
            0.05 * completeness_score +
            0.40 * sharpness_score +
            0.35 * coverage_score
        )

        return total_score, {
            'size': size_score,
            'shape': shape_score,
            'completeness': completeness_score,
            'sharpness': sharpness_score,
            'coverage': coverage_score,
            'area_ratio': area_ratio,
            'solidity': solidity,
            'extent': extent
        }

    def _refine_contour_halfmax(self, warped_image, contour):
        """
        Re-extract the object contour at the half-maximum intensity between
        the object's own gray level and the local background level.

        A fixed threshold places the edge somewhere inside the blur halo
        around the object, biasing every dimension outward by a fraction of
        the blur width. The 50% crossing point of a blurred step edge is at
        the true edge position, so thresholding at the midpoint between
        object and background intensity removes that bias.

        Returns the refined contour, or the original if refinement fails.
        """
        gray = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)

        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [contour], -1, 255, -1)

        # Sample the object interior and a nearby background ring, staying
        # clear of the blurred edge itself
        interior = cv2.erode(mask, np.ones((9, 9), np.uint8))
        near = cv2.dilate(mask, np.ones((9, 9), np.uint8))
        far = cv2.dilate(mask, np.ones((41, 41), np.uint8))
        background_ring = (far > 0) & (near == 0)

        if interior.sum() == 0 or background_ring.sum() == 0:
            return contour

        obj_level = float(np.median(gray[interior > 0]))
        bg_level = float(np.median(gray[background_ring]))
        if bg_level - obj_level < 20:  # too little contrast to trust
            return contour

        mid = (obj_level + bg_level) / 2.0
        _, binary = cv2.threshold(gray, mid, 255, cv2.THRESH_BINARY_INV)
        # Only consider the neighbourhood of the original detection
        binary[far == 0] = 0

        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return contour

        # Pick the refined contour that best overlaps the original
        best, best_overlap = contour, 0
        for c in contours:
            m = np.zeros_like(gray)
            cv2.drawContours(m, [c], -1, 255, -1)
            overlap = int(np.count_nonzero((m > 0) & (mask > 0)))
            if overlap > best_overlap:
                best, best_overlap = c, overlap

        if best_overlap == 0:
            return contour

        # Sanity guard: refinement corrects sub-mm blur bias, so the area
        # should barely change. A large growth means it latched onto a
        # shadow or neighbouring region - keep the original instead.
        orig_area = max(cv2.contourArea(contour), 1)
        if not (0.6 < cv2.contourArea(best) / orig_area < 1.3):
            return contour

        return best

    def measure_object(self, warped_image, threshold=200, debug=False, use_convex_hull=False):
        """
        Measure the object in the calibrated image using adaptive multi-threshold segmentation

        Args:
            warped_image: Calibrated/warped image (inner window only)
            threshold: Base threshold (used as fallback, but multi-threshold is primary)
            debug: Show debug visualizations
            use_convex_hull: If True, compute convex hull of all detected parts (for hollow objects)

        Returns:
            Dictionary with measurements
        """
        h, w = warped_image.shape[:2]
        total_area = h * w

        # Small margin to ignore warp artifacts at the window edge
        edge_margin = max(5, int(min(h, w) * 0.01))
        min_object_area = total_area * 0.001
        max_object_area = total_area * 0.6
        max_width = w * 0.95
        max_height = h * 0.95

        # ADAPTIVE MULTI-THRESHOLD SEGMENTATION
        # Thresholds relative to the actual background brightness, so dim or
        # bright photos both work. Median over the window is dominated by
        # background pixels.
        gray = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
        background = float(np.median(gray))
        thresholds_to_try = sorted({max(30, int(background * f))
                                    for f in (0.50, 0.62, 0.74, 0.85, 0.93)})

        # Precompute gradient magnitude for boundary-sharpness scoring
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gradient_mag = cv2.magnitude(gx, gy)

        # Canny edge map for coverage scoring, ignoring warp artifacts at
        # the window border
        edges = cv2.Canny(gray, 50, 150)
        edges[:edge_margin, :] = 0
        edges[-edge_margin:, :] = 0
        edges[:, :edge_margin] = 0
        edges[:, -edge_margin:] = 0

        all_candidates = []

        for thresh in thresholds_to_try:
            binary = self.segment_object(warped_image, threshold=thresh, debug=False)
            contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                x, y, cw, ch = cv2.boundingRect(contour)

                is_away_from_edges = (x > edge_margin and y > edge_margin and
                                     x + cw < w - edge_margin and y + ch < h - edge_margin)
                is_reasonable_size = (min_object_area < area < max_object_area)
                is_not_full_frame = (cw < max_width and ch < max_height)

                if is_away_from_edges and is_reasonable_size and is_not_full_frame:
                    score, metrics = self._score_contour(contour, gradient_mag,
                                                         edges, warped_image.shape)

                    all_candidates.append({
                        'contour': contour,
                        'threshold': thresh,
                        'score': score,
                        'metrics': metrics,
                        'area': area
                    })

        if len(all_candidates) == 0:
            return {"error": "No object detected at any threshold (tried 170-245)"}

        # CONVEX HULL MODE: For objects with large hollow interiors (e.g., pliers, scissors)
        if use_convex_hull and len(all_candidates) >= 2:
            all_points = []
            for candidate in all_candidates:
                all_points.extend(candidate['contour'].reshape(-1, 2))

            all_points = np.array(all_points, dtype=np.float32)
            hull = cv2.convexHull(all_points)
            object_contour = hull.reshape((-1, 1, 2)).astype(np.int32)

            if debug:
                print(f"\n  Convex Hull Mode: Combined {len(all_candidates)} parts into single hull")
        else:
            best_candidate = max(all_candidates, key=lambda x: x['score'])
            object_contour = self._refine_contour_halfmax(warped_image,
                                                          best_candidate['contour'])

        if debug and not use_convex_hull:
            print(f"\nAdaptive Segmentation Results:")
            print(f"  Tried {len(thresholds_to_try)} thresholds, found {len(all_candidates)} candidates")
            print(f"  Best: threshold={best_candidate['threshold']}, score={best_candidate['score']:.3f}")

        # Calculate measurements in pixels
        perimeter_pixels = cv2.arcLength(object_contour, closed=True)
        area_pixels = cv2.contourArea(object_contour)

        x, y, bw, bh = cv2.boundingRect(object_contour)

        scale = 10  # pixels per mm (from warp_to_calibrated_view)

        measurements = {
            "perimeter_mm": perimeter_pixels / scale,
            "area_mm2": area_pixels / (scale * scale),
            "bounding_box_width_mm": bw / scale,
            "bounding_box_height_mm": bh / scale,
            "contour": object_contour,
            "bounding_box": (x, y, bw, bh)
        }

        if debug:
            debug_img = warped_image.copy()
            cv2.drawContours(debug_img, [object_contour], -1, (0, 255, 0), 2)
            cv2.rectangle(debug_img, (x, y), (x + bw, y + bh), (255, 0, 0), 2)
            cv2.imshow('Object Measurement', debug_img)

        return measurements


def main():
    parser = argparse.ArgumentParser(description='Measure objects using calibration frame')
    parser.add_argument('image', help='Path to image file')
    parser.add_argument('--width', type=float, default=DEFAULT_INNER_WIDTH_MM,
                       help=f'Inner window width in mm (default: {DEFAULT_INNER_WIDTH_MM})')
    parser.add_argument('--height', type=float, default=DEFAULT_INNER_HEIGHT_MM,
                       help=f'Inner window height in mm (default: {DEFAULT_INNER_HEIGHT_MM})')
    parser.add_argument('--debug', action='store_true', help='Show debug visualizations')
    parser.add_argument('--threshold', type=int, default=200,
                       help='Segmentation threshold (default: 200, lower for darker objects)')
    parser.add_argument('--convex-hull', action='store_true',
                       help='Use convex hull mode for objects with hollow interiors (pliers, scissors, etc.)')
    parser.add_argument('--save-output',
                       help='Path for the annotated output image '
                            '(default: output/<image>_measured.png in the repo)')

    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Could not load image from {args.image}")
        return

    print(f"Loaded image: {image.shape[1]}x{image.shape[0]} pixels")

    frame = CalibrationFrame(inner_width_mm=args.width, inner_height_mm=args.height)

    print("\nCalibrating...")
    if not frame.calibrate(image, debug=args.debug):
        print("Calibration failed!")
        return

    print("\nWarping to calibrated view...")
    warped = frame.warp_to_calibrated_view(image)

    if args.debug:
        cv2.imshow('Warped Calibrated View', warped)

    print("\nMeasuring object...")
    measurer = ObjectMeasurer(frame)
    measurements = measurer.measure_object(warped, threshold=args.threshold,
                                          debug=args.debug, use_convex_hull=args.convex_hull)

    if "error" in measurements:
        print(f"Error: {measurements['error']}")
        print("Try adjusting the --threshold parameter")
    else:
        print("\n" + "="*50)
        print("FINAL MEASUREMENTS")
        print("="*50)
        print(f"Perimeter:     {measurements['perimeter_mm']:.2f} mm")
        print(f"Area:          {measurements['area_mm2']:.2f} mm²")
        print(f"Bounding Box:  {measurements['bounding_box_width_mm']:.2f} x {measurements['bounding_box_height_mm']:.2f} mm")
        print("="*50)

    if "contour" in measurements:
        if args.save_output:
            out_path = Path(args.save_output)
        else:
            OUTPUT_DIR.mkdir(exist_ok=True)
            out_path = OUTPUT_DIR / f"{Path(args.image).stem}_measured.png"

        output = warped.copy()
        cv2.drawContours(output, [measurements['contour']], -1, (0, 255, 0), 3)
        x, y, w, h = measurements['bounding_box']
        cv2.rectangle(output, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.imwrite(str(out_path), output)
        print(f"\nSaved annotated output to: {out_path}")

    if args.debug:
        print("\nPress any key to close windows...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
