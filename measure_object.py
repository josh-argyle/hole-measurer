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
DEFAULT_BORDER_MM = 10.0


class CalibrationFrame:
    """Detects the black frame's inner window and calibrates the view"""

    def __init__(self, inner_width_mm=DEFAULT_INNER_WIDTH_MM,
                 inner_height_mm=DEFAULT_INNER_HEIGHT_MM,
                 border_mm=DEFAULT_BORDER_MM):
        """
        Args:
            inner_width_mm: Width of the frame's inner opening (mm)
            inner_height_mm: Height of the frame's inner opening (mm)
            border_mm: Width of the black border (used by the outer-edge
                calibration fallback when the inner window is obstructed)
        """
        self.inner_width_mm = inner_width_mm
        self.inner_height_mm = inner_height_mm
        self.border_mm = border_mm
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

    def detect_outer_edge(self, image, debug=False):
        """
        Fallback: find the OUTER edge of the black frame band. Works when
        the inner window is obstructed (object/paper overlapping the inner
        edge, reflections bridging inside to outside), as long as the frame
        contrasts with the surface it sits on.

        Returns:
            (4, 2) float32 array of outer corners ordered TL, TR, BR, BL
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # The frame is near-black; threshold well below the general scene
        # brightness so mid-tone surfaces (cardboard, wood) stay excluded
        p5, p95 = np.percentile(gray, [5, 95])
        _, dark = cv2.threshold(gray, p5 + 0.25 * (p95 - p5), 255, cv2.THRESH_BINARY_INV)
        # Seal small bright streaks (glossy print reflections)
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

        contours, hierarchy = cv2.findContours(dark, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("No dark regions found - is the frame in the image?")

        h, w = gray.shape
        image_area = h * w

        # The frame band is a large dark region that encloses a large hole
        # (the window) and does not touch the image border
        best, best_hole = None, 0
        for i, (c, hier) in enumerate(zip(contours, hierarchy[0])):
            if hier[3] != -1:
                continue
            x, y, cw, ch = cv2.boundingRect(c)
            if x <= 1 or y <= 1 or x + cw >= w - 1 or y + ch >= h - 1:
                continue
            if cv2.contourArea(c) < 0.05 * image_area:
                continue
            hole_area = 0
            child = hier[2]
            while child != -1:
                hole_area = max(hole_area, cv2.contourArea(contours[child]))
                child = hierarchy[0][child][0]
            if hole_area > best_hole:
                best, best_hole = c, hole_area

        if best is None or best_hole < 0.05 * image_area:
            raise ValueError("Could not find the frame band - the black frame "
                             "must be fully in view and contrast with the surface")

        peri = cv2.arcLength(best, True)
        approx = cv2.approxPolyDP(best, 0.02 * peri, True)
        if len(approx) != 4:
            approx = cv2.boxPoints(cv2.minAreaRect(best))

        rough = self._order_corners(np.array(approx, dtype=np.float32))
        corners = self._corners_from_edge_lines(best, rough)

        if debug:
            print(f"Outer frame corners (TL, TR, BR, BL): {corners.tolist()}")

        return corners

    def calibrate(self, image, debug=False):
        """
        Detect the frame and compute the pixel->mm homography.

        Tries the inner window first; falls back to the frame's outer edge
        if the window can't be found cleanly.

        Returns:
            True if calibration succeeded, False otherwise
        """
        b = self.border_mm
        try:
            corners = self.detect_inner_window(image, debug=debug)
            known = np.array([
                [0, 0],
                [self.inner_width_mm, 0],
                [self.inner_width_mm, self.inner_height_mm],
                [0, self.inner_height_mm],
            ], dtype=np.float32)
        except ValueError as inner_err:
            try:
                corners = self.detect_outer_edge(image, debug=debug)
                print("Note: inner window not usable "
                      f"({inner_err}); calibrated from the frame's outer edge")
                known = np.array([
                    [-b, -b],
                    [self.inner_width_mm + b, -b],
                    [self.inner_width_mm + b, self.inner_height_mm + b],
                    [-b, self.inner_height_mm + b],
                ], dtype=np.float32)
            except ValueError as e:
                print(f"Error: {e}")
                return False

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

    def segment_object(self, warped_image, delta=40, debug=False):
        """
        Segment the object as pixels deviating from the background level,
        in either direction - handles dark objects on light backgrounds AND
        shiny/light objects on darker backgrounds.

        Args:
            delta: Minimum absolute difference from the background gray level

        Returns:
            Binary mask of the object
        """
        gray = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY).astype(np.int16)
        background = float(np.median(gray))

        binary = (np.abs(gray - background) > delta).astype(np.uint8) * 255

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
        if abs(bg_level - obj_level) < 20:  # too little contrast to trust
            return contour

        mid = (obj_level + bg_level) / 2.0
        # Select the object side of the midpoint, whichever polarity
        if obj_level < bg_level:
            _, binary = cv2.threshold(gray, mid, 255, cv2.THRESH_BINARY_INV)
        else:
            _, binary = cv2.threshold(gray, mid, 255, cv2.THRESH_BINARY)
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

        # Sanity guard: refinement corrects sub-mm blur bias, so neither the
        # area nor the bounding box should change much. Growth means it
        # latched onto a shadow; shrinkage means it dropped part of a
        # mixed-brightness object (e.g. shiny metal with dark shading).
        orig_area = max(cv2.contourArea(contour), 1)
        if not (0.75 < cv2.contourArea(best) / orig_area < 1.25):
            return contour
        _, _, ow, oh = cv2.boundingRect(contour)
        _, _, nw, nh = cv2.boundingRect(best)
        if abs(nw - ow) > 20 or abs(nh - oh) > 20:  # >2mm bbox change
            return contour

        return best

    def _extend_with_edges(self, edges, contour, image_shape):
        """
        Reattach faint thin parts (e.g. tweezer tips) that thresholding
        missed. Such parts are too close to the background level to
        segment, but their outlines still show in the Canny edge map.
        Take edge pixels connected to the object and merge them in.

        Returns the extended contour, or the original if nothing to add.
        """
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)

        k3 = np.ones((3, 3), np.uint8)
        combined = cv2.bitwise_or(mask, cv2.dilate(edges, k3))

        # Keep only the connected component(s) touching the object
        n, labels = cv2.connectedComponents(combined)
        object_labels = np.unique(labels[mask > 0])
        keep = np.isin(labels, object_labels[object_labels != 0])
        extended = np.where(keep, np.uint8(255), np.uint8(0))

        # Every object's own boundary contributes a thin ring of edge pixels
        # just outside the mask - merging that in would inflate all
        # measurements. A genuine missed part (a tweezer tip) reaches far
        # beyond the boundary. Keep only added components that extend well
        # away from the original mask.
        added = cv2.bitwise_and(extended, cv2.bitwise_not(mask))
        dist_outside = cv2.distanceTransform(cv2.bitwise_not(mask), cv2.DIST_L2, 5)
        n_add, add_labels = cv2.connectedComponents(added)
        result = mask.copy()
        for label in range(1, n_add):
            comp = (add_labels == label)
            # Must reach >1mm beyond the boundary (not just the edge ring)
            if dist_outside[comp].max() <= 10:
                continue
            # Must be a THIN structure (tip, arm): interior never more than
            # ~2mm from its own boundary. Fat additions (shadow lobes,
            # neighbouring blobs) are rejected.
            comp_mask = comp.astype(np.uint8) * 255
            thickness = cv2.distanceTransform(comp_mask, cv2.DIST_L2, 5).max()
            if thickness <= 20:
                result[comp] = 255

        if np.array_equal(result, mask):
            return contour

        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        contours, _ = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return contour
        best = max(contours, key=cv2.contourArea)

        # Guard: even thin extensions shouldn't multiply the area many times
        # over - that means the edge map bridged across the scene.
        orig_area = max(cv2.contourArea(contour), 1)
        if cv2.contourArea(best) / orig_area > 3.0:
            return contour

        return best

    def _smooth_contour(self, contour, strength=2.5):
        """
        Smooth a contour so straight edges draw straight.

        strength sets both the Gaussian sigma and the simplification
        tolerance; 0 disables smoothing entirely.

        Two stages: a circular Gaussian filter along the contour removes
        pixel-level jaggies from thresholding, then Douglas-Peucker
        simplification collapses near-straight runs into single segments
        while preserving genuine corners and curves.

        Also makes the perimeter honest - stair-stepped pixel edges
        inflate arc length by several percent.
        """
        if strength <= 0:
            return contour
        sigma = strength
        epsilon = max(1.0, strength * 0.8)

        pts = contour.reshape(-1, 2).astype(np.float64)
        if len(pts) < 12:
            return contour

        # Resample to uniform 1px spacing along the closed loop - contour
        # vertices are unevenly spaced (straight runs keep only endpoints),
        # and Gaussian filtering assumes uniform samples
        closed = np.vstack([pts, pts[:1]])
        seglen = np.linalg.norm(np.diff(closed, axis=0), axis=1)
        arclen = np.concatenate([[0], np.cumsum(seglen)])
        total = arclen[-1]
        if total < 12:
            return contour
        s = np.arange(0, total, 1.0)
        pts = np.stack([np.interp(s, arclen, closed[:, 0]),
                        np.interp(s, arclen, closed[:, 1])], axis=1)
        n = len(pts)

        ksize = max(3, int(sigma * 6) | 1)
        half = ksize // 2
        kernel = np.exp(-0.5 * ((np.arange(ksize) - half) / sigma) ** 2)
        kernel /= kernel.sum()

        # Find corners and tips: points where the contour turns sharply
        # within a small support window. These are anchored - a Gaussian
        # blindly rounds them inward, visibly shortening pointed objects.
        k = max(3, int(round(sigma * 2)))
        v1 = pts - np.roll(pts, k, axis=0)
        v2 = np.roll(pts, -k, axis=0) - pts
        dot = (v1 * v2).sum(axis=1)
        norm = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
        turn = np.degrees(np.arccos(np.clip(dot / np.maximum(norm, 1e-9), -1, 1)))
        corner_mask = turn > 50

        def smooth_open(seg):
            """Gaussian-smooth an open segment, endpoints held fixed
            (antisymmetric reflection padding preserves them exactly)"""
            if len(seg) < 2 * half + 3:
                return seg
            front = 2 * seg[0] - seg[half:0:-1]
            back = 2 * seg[-1] - seg[-2:-2 - half:-1]
            ext = np.vstack([front, seg, back])
            xs = np.convolve(ext[:, 0], kernel, mode='valid')
            ys = np.convolve(ext[:, 1], kernel, mode='valid')
            return np.stack([xs, ys], axis=1)

        if not corner_mask.any():
            # No corners (e.g. a disc): circular smoothing of the whole loop
            ext = np.vstack([pts[-half:], pts, pts[:half]])
            xs = np.convolve(ext[:, 0], kernel, mode='valid')
            ys = np.convolve(ext[:, 1], kernel, mode='valid')
            smooth = np.stack([xs, ys], axis=1)
        elif corner_mask.all():
            return contour
        else:
            # One anchor per contiguous corner run: its sharpest point.
            # Rotate so index 0 is not inside a run, keeping runs contiguous.
            start = int(np.argmin(corner_mask))
            rolled = np.roll(corner_mask, -start)
            turn_rolled = np.roll(turn, -start)
            anchors = []
            i = 0
            while i < n:
                if rolled[i]:
                    j = i
                    while j < n and rolled[j]:
                        j += 1
                    run = np.arange(i, j)
                    anchors.append((int(run[np.argmax(turn_rolled[run])]) + start) % n)
                    i = j
                else:
                    i += 1
            anchors.sort()

            # Smooth each stretch between consecutive anchors independently
            pieces = []
            m = len(anchors)
            for idx in range(m):
                a, b = anchors[idx], anchors[(idx + 1) % m]
                seg = pts[a:b + 1] if b > a else np.vstack([pts[a:], pts[:b + 1]])
                pieces.append(smooth_open(seg)[:-1])  # drop dup anchor
            smooth = np.vstack(pieces)

        approx = cv2.approxPolyDP(smooth.astype(np.float32).reshape(-1, 1, 2),
                                  epsilon, True)
        if len(approx) < 3:
            return contour
        return approx.astype(np.int32)

    def measure_object(self, warped_image, threshold=200, debug=False, use_convex_hull=False,
                       smooth=2.5):
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

        # ADAPTIVE MULTI-DELTA SEGMENTATION
        # Try several contrast levels relative to the background so faint
        # and strong objects both segment; scoring picks the best candidate.
        gray = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
        background = float(np.median(gray))
        contrast_range = max(background, 255 - background)
        deltas_to_try = sorted({max(12, int(contrast_range * f))
                                for f in (0.07, 0.15, 0.26, 0.38, 0.50)})

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

        for delta in deltas_to_try:
            binary = self.segment_object(warped_image, delta=delta, debug=False)
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
                        'delta': delta,
                        'score': score,
                        'metrics': metrics,
                        'area': area
                    })

        if len(all_candidates) == 0:
            return {"error": "No object detected at any contrast level - "
                             "check the object contrasts with the background"}

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
            object_contour = self._extend_with_edges(edges, object_contour,
                                                     warped_image.shape)
            object_contour = self._smooth_contour(object_contour, strength=smooth)

        if debug and not use_convex_hull:
            print(f"\nAdaptive Segmentation Results:")
            print(f"  Tried {len(deltas_to_try)} contrast levels, found {len(all_candidates)} candidates")
            print(f"  Best: delta={best_candidate['delta']}, score={best_candidate['score']:.3f}")

        # Calculate measurements in pixels
        perimeter_pixels = cv2.arcLength(object_contour, closed=True)
        area_pixels = cv2.contourArea(object_contour)

        x, y, bw, bh = cv2.boundingRect(object_contour)

        # Oriented bounding box: the smallest rectangle at any angle, so the
        # reported length/width follow the object, not the photo axes
        (rcx, rcy), (rw, rh), rangle = cv2.minAreaRect(object_contour)
        length_px, width_px = max(rw, rh), min(rw, rh)
        rect_box = cv2.boxPoints(((rcx, rcy), (rw, rh), rangle))

        scale = 10  # pixels per mm (from warp_to_calibrated_view)

        measurements = {
            "perimeter_mm": perimeter_pixels / scale,
            "area_mm2": area_pixels / (scale * scale),
            "length_mm": length_px / scale,
            "width_mm": width_px / scale,
            "bounding_box_width_mm": bw / scale,
            "bounding_box_height_mm": bh / scale,
            "contour": object_contour,
            "bounding_box": (x, y, bw, bh),
            "oriented_box": rect_box,
        }

        if debug:
            debug_img = warped_image.copy()
            cv2.drawContours(debug_img, [object_contour], -1, (0, 255, 0), 2)
            cv2.rectangle(debug_img, (x, y), (x + bw, y + bh), (255, 0, 0), 2)
            cv2.imshow('Object Measurement', debug_img)

        return measurements


def export_outline(measurements, image_height_px, base_path):
    """
    Export the measured outline as CAD-ready vector files in millimetres:
    a DXF (R12 closed polyline, importable by Fusion 360 / FreeCAD /
    Tinkercad for extruding) and an SVG. Returns (dxf_path, svg_path).
    """
    scale = 10.0  # warped px per mm
    pts = measurements['contour'].reshape(-1, 2).astype(np.float64)
    # DXF is Y-up; flip so the part isn't mirrored and stays positive
    mm = [(x / scale, (image_height_px - y) / scale) for x, y in pts]

    dxf_path = str(base_path) + '_outline.dxf'
    lines = ['0', 'SECTION', '2', 'ENTITIES',
             '0', 'POLYLINE', '8', '0', '66', '1', '70', '1']
    for x, y in mm:
        lines += ['0', 'VERTEX', '8', '0',
                  '10', f'{x:.3f}', '20', f'{y:.3f}']
    lines += ['0', 'SEQEND', '0', 'ENDSEC', '0', 'EOF']
    with open(dxf_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    svg_path = str(base_path) + '_outline.svg'
    xs = [p[0] for p in mm]
    ys = [image_height_px / scale - p[1] for p in mm]  # SVG is Y-down
    w = max(xs) - min(xs)
    h = max(ys) - min(ys)
    d = 'M ' + ' L '.join(f'{x - min(xs):.3f},{y - min(ys):.3f}'
                          for x, y in zip(xs, ys)) + ' Z'
    with open(svg_path, 'w') as f:
        f.write(
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{w:.2f}mm" height="{h:.2f}mm" '
            f'viewBox="0 0 {w:.3f} {h:.3f}">\n'
            f'  <path d="{d}" fill="none" stroke="black" stroke-width="0.2"/>\n'
            f'</svg>\n')

    return dxf_path, svg_path


def annotate_measurements(warped_image, measurements):
    """
    Draw the contour, bounding box, and dimension labels on each edge
    of the bounding box. Returns the annotated image.
    """
    out = warped_image.copy()
    h_img, w_img = out.shape[:2]

    contour = measurements['contour']
    box = measurements['oriented_box'].astype(np.float32)

    # Rotate the view so the object's long axis is vertical, if the rotated
    # object (with room for dimension labels) still fits inside the window
    e0, e1 = box[1] - box[0], box[2] - box[1]
    v = e0 if np.linalg.norm(e0) >= np.linalg.norm(e1) else e1
    theta = np.degrees(np.arctan2(v[1], v[0]))
    a = (theta - 90) % 180
    if a > 90:
        a -= 180
    if abs(a) > 2:
        center = tuple(box.mean(axis=0))
        M = cv2.getRotationMatrix2D(center, a, 1.0)
        # Verify direction: the long axis must end up vertical
        R = M[:, :2]
        v_rot = R @ v
        if abs(v_rot[0]) > abs(v_rot[1]):
            M = cv2.getRotationMatrix2D(center, -a, 1.0)

        pts = contour.reshape(-1, 2).astype(np.float32)
        pts_rot = pts @ M[:, :2].T + M[:, 2]
        margin = 40  # room for dimension lines and labels
        if (pts_rot.min(axis=0) > margin).all() and \
           (pts_rot[:, 0].max() < w_img - margin) and \
           (pts_rot[:, 1].max() < h_img - margin):
            out = cv2.warpAffine(out, M, (w_img, h_img),
                                 borderMode=cv2.BORDER_REPLICATE)
            contour = pts_rot.reshape(-1, 1, 2).astype(np.int32)
            box = box @ M[:, :2].T + M[:, 2]

    cv2.drawContours(out, [contour], -1, (0, 255, 0), 3)

    def dotted_polyline(pts, color, thickness=2, dash=5, gap=7):
        """Draw a closed polyline as dots (OpenCV has no dashed primitive)"""
        loop = np.vstack([pts, pts[:1]]).astype(np.float64)
        for p1, p2 in zip(loop[:-1], loop[1:]):
            seg = p2 - p1
            length = np.linalg.norm(seg)
            if length < 1:
                continue
            direction = seg / length
            pos = 0.0
            while pos < length:
                a = p1 + direction * pos
                b = p1 + direction * min(pos + dash, length)
                cv2.line(out, tuple(a.astype(int)), tuple(b.astype(int)),
                         color, thickness)
                pos += dash + gap

    dotted_polyline(box, (255, 0, 0))

    font = cv2.FONT_HERSHEY_SIMPLEX
    fscale = max(0.6, min(w_img, h_img) / 1200.0)
    fthick = max(1, int(round(fscale * 2)))
    blue = (255, 0, 0)

    def label(text, cx, cy):
        """Draw text centered at (cx, cy) on a white pill, clamped inside"""
        (tw, th), base = cv2.getTextSize(text, font, fscale, fthick)
        cx = int(np.clip(cx, tw // 2 + 6, w_img - tw // 2 - 6))
        cy = int(np.clip(cy, th + 6, h_img - base - 6))
        cv2.rectangle(out, (cx - tw // 2 - 5, cy - th - 5),
                      (cx + tw // 2 + 5, cy + base + 5), (255, 255, 255), -1)
        cv2.rectangle(out, (cx - tw // 2 - 5, cy - th - 5),
                      (cx + tw // 2 + 5, cy + base + 5), blue, 1)
        cv2.putText(out, text, (cx - tw // 2, cy), font, fscale, blue, fthick)

    center = box.mean(axis=0)

    def dimension_line(p1, p2, text):
        """Dimension line parallel to edge p1-p2, offset away from center,
        with end ticks and a centered label"""
        edge = p2 - p1
        elen = np.linalg.norm(edge)
        if elen < 1:
            return
        normal = np.array([-edge[1], edge[0]]) / elen
        mid = (p1 + p2) / 2
        if np.dot(mid - center, normal) < 0:
            normal = -normal  # point outward
        gap, tick = 20, 8
        a = p1 + normal * gap
        b = p2 + normal * gap
        cv2.line(out, tuple(a.astype(int)), tuple(b.astype(int)), blue, 2)
        for p in (a, b):
            t1 = p - normal * tick
            t2 = p + normal * tick
            cv2.line(out, tuple(t1.astype(int)), tuple(t2.astype(int)), blue, 2)
        lp = mid + normal * (gap + 26)
        label(text, lp[0], lp[1])

    # boxPoints returns points in order; adjacent edges alternate between
    # the two side lengths. Label one long edge and one short edge.
    e0_len = np.linalg.norm(box[1] - box[0])
    e1_len = np.linalg.norm(box[2] - box[1])
    length_text = f"{measurements['length_mm']:.1f} mm"
    width_text = f"{measurements['width_mm']:.1f} mm"
    if e0_len >= e1_len:
        dimension_line(box[0], box[1], length_text)
        dimension_line(box[1], box[2], width_text)
    else:
        dimension_line(box[1], box[2], length_text)
        dimension_line(box[0], box[1], width_text)

    # Summary banner
    label(f"Perimeter {measurements['perimeter_mm']:.1f} mm   "
          f"Area {measurements['area_mm2']:.0f} mm2", w_img // 2, 14)

    return out


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
    parser.add_argument('--json', action='store_true',
                       help='Print machine-readable JSON result as the last line')
    parser.add_argument('--smooth', type=float, default=2.5,
                       help='Contour smoothing strength (0 = off, default 2.5)')

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
        if args.json:
            import json
            print(json.dumps({"ok": False,
                              "error": "Calibration failed - make sure the whole "
                                       "black frame is visible in the photo"}))
        return

    print("\nWarping to calibrated view...")
    warped = frame.warp_to_calibrated_view(image)

    if args.debug:
        cv2.imshow('Warped Calibrated View', warped)

    print("\nMeasuring object...")
    measurer = ObjectMeasurer(frame)
    measurements = measurer.measure_object(warped, threshold=args.threshold,
                                          debug=args.debug, use_convex_hull=args.convex_hull,
                                          smooth=args.smooth)

    if "error" in measurements:
        print(f"Error: {measurements['error']}")
        print("Try adjusting the --threshold parameter")
    else:
        print("\n" + "="*50)
        print("FINAL MEASUREMENTS")
        print("="*50)
        print(f"Length x Width: {measurements['length_mm']:.2f} x {measurements['width_mm']:.2f} mm (oriented)")
        print(f"Perimeter:      {measurements['perimeter_mm']:.2f} mm")
        print(f"Area:           {measurements['area_mm2']:.2f} mm²")
        print(f"Bounding Box:   {measurements['bounding_box_width_mm']:.2f} x {measurements['bounding_box_height_mm']:.2f} mm (photo axes)")
        print("="*50)

    out_path = None
    if "contour" in measurements:
        if args.save_output:
            out_path = Path(args.save_output)
        else:
            OUTPUT_DIR.mkdir(exist_ok=True)
            out_path = OUTPUT_DIR / f"{Path(args.image).stem}_measured.png"

        output = annotate_measurements(warped, measurements)
        cv2.imwrite(str(out_path), output)
        print(f"\nSaved annotated output to: {out_path}")

        base = out_path.parent / out_path.stem.replace('_measured', '')
        dxf_path, svg_path = export_outline(measurements, warped.shape[0], base)
        print(f"Saved CAD outline to: {dxf_path} and {svg_path}")

    if args.json:
        import json
        if "error" in measurements:
            payload = {"ok": False, "error": measurements["error"]}
        else:
            payload = {
                "ok": True,
                "length_mm": round(measurements['length_mm'], 2),
                "width_mm": round(measurements['width_mm'], 2),
                "perimeter_mm": round(measurements['perimeter_mm'], 2),
                "area_mm2": round(measurements['area_mm2'], 2),
                "mm_per_pixel": round(frame.mm_per_pixel, 5),
                "output_image": str(out_path),
                "dxf": dxf_path,
                "svg": svg_path,
            }
        print(json.dumps(payload))

    if args.debug:
        print("\nPress any key to close windows...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
