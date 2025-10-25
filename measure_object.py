#!/usr/bin/env python3
"""
Calibration Frame Object Measurement System
Measures 2D objects placed within a calibrated frame with fiducial markers
"""

import cv2
import numpy as np
import argparse
import json
from pathlib import Path


class CalibrationFrame:
    """Handles detection and calibration of the frame with hole markers"""
    
    def __init__(self, frame_width_mm=180.0, frame_height_mm=101.25):
        """
        Initialize with frame dimensions in millimeters
        
        Args:
            frame_width_mm: Width of the black frame (outer edge to outer edge)
            frame_height_mm: Height of the black frame (outer edge to outer edge)
        """
        self.frame_width_mm = frame_width_mm
        self.frame_height_mm = frame_height_mm
        self.known_positions = self._generate_known_positions()
        self.homography_matrix = None
        self.mm_per_pixel = None
        
    def _generate_known_positions(self):
        """
        Generate the known positions of all holes in mm coordinates.
        Origin at top-left corner.
        
        Pattern:
        - 4 corners
        - Top edge: 1/2, 1/4, 1/8, 1/16, 1/32, 1/64 from left
        - Left edge: 1/2, 1/4, 1/8, 1/16, 1/32 from top
        """
        positions = []
        w = self.frame_width_mm
        h = self.frame_height_mm
        
        # 4 corners (in order: top-left, top-right, bottom-right, bottom-left)
        corners = [
            (0, 0, 'top_left'),
            (w, 0, 'top_right'),
            (w, h, 'bottom_right'),
            (0, h, 'bottom_left')
        ]
        positions.extend(corners)
        
        # Top edge holes (left to right)
        top_fractions = [1/2, 1/4, 1/8, 1/16, 1/32, 1/64]
        for frac in top_fractions:
            positions.append((w * frac, 0, f'top_{frac}'))
        
        # Left edge holes (top to bottom)
        left_fractions = [1/2, 1/4, 1/8, 1/16, 1/32]
        for frac in left_fractions:
            positions.append((0, h * frac, f'left_{frac}'))
        
        return positions
    
    def detect_holes(self, image, debug=False):
        """
        Detect all white circular holes in the image

        Args:
            image: Input image (BGR)
            debug: If True, show intermediate detection steps

        Returns:
            List of (x, y, area) tuples for detected hole centers and sizes
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Threshold to find white holes
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        if debug:
            cv2.imshow('Binary Threshold', binary)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours to find circular holes
        holes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 50:  # Skip very small contours (noise)
                continue

            # Check circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)

            if circularity > 0.7:  # Reasonably circular
                # Get center
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    cx = M['m10'] / M['m00']
                    cy = M['m01'] / M['m00']
                    holes.append((cx, cy, area))

        if debug:
            debug_img = image.copy()
            for (x, y, _) in holes:
                cv2.circle(debug_img, (int(x), int(y)), 5, (0, 255, 0), -1)
            cv2.imshow('All Detected Circular Holes', debug_img)
            print(f"Detected {len(holes)} circular holes (before outlier filtering)")

        return holes
    
    def match_holes_to_pattern(self, detected_holes, n_iterations=1000, debug=False):
        """
        Match detected holes to known pattern using RANSAC outlier rejection

        This method uses RANSAC to robustly identify which detected holes
        correspond to the calibration frame, rejecting spurious detections
        from objects (like wrench holes).

        Args:
            detected_holes: List of (x, y, area) tuples from detect_holes()
            n_iterations: Number of RANSAC iterations (default 1000)
            debug: If True, print diagnostic information

        Returns:
            Tuple of (detected_points, known_points, inlier_mask, outlier_info)
        """
        # Extract just x,y coordinates and sizes
        if len(detected_holes) == 0:
            raise ValueError("No holes detected")

        if isinstance(detected_holes[0], tuple) and len(detected_holes[0]) == 3:
            hole_positions = np.array([(x, y) for x, y, _ in detected_holes], dtype=np.float32)
            hole_sizes = np.array([area for _, _, area in detected_holes])
        else:
            hole_positions = np.array(detected_holes, dtype=np.float32)
            hole_sizes = None

        n_detected = len(hole_positions)

        if n_detected < 4:
            raise ValueError(f"Need at least 4 holes, found {n_detected}")

        # Get all 15 known hole positions
        known_positions = np.array([[x, y] for x, y, _ in self.known_positions], dtype=np.float32)

        # Known corner positions (first 4 in known_positions)
        known_corners = known_positions[:4]

        # FIRST: Identify the 4 actual corner holes using extrema from ALL holes
        # These should be the most extreme points
        tl_idx = np.argmin(hole_positions.sum(axis=1))
        br_idx = np.argmax(hole_positions.sum(axis=1))
        tr_idx = np.argmax(hole_positions[:, 0] - hole_positions[:, 1])
        bl_idx = np.argmin(hole_positions[:, 0] - hole_positions[:, 1])

        corner_indices = np.array([tl_idx, tr_idx, br_idx, bl_idx])
        corner_positions = hole_positions[corner_indices]

        if debug:
            print(f"Identified corners at indices: {corner_indices}")
            print(f"Corner positions: {corner_positions.tolist()}")

        # RANSAC: Try different hypotheses for which detected hole matches which known position
        # But always use the identified corners as corners
        best_inliers = None
        best_inlier_count = 0
        best_homography = None
        best_residuals = None

        for iteration in range(n_iterations):
            # Use identified corners, but try different permutations of assignment
            # There are 4! = 24 possible ways to assign 4 corners, but most are invalid
            # We'll stick with the geometric assignment: tl->TL, tr->TR, br->BR, bl->BL
            # Instead, add some jitter/try nearby holes as corners

            if iteration < 100:
                # First 100 iterations: use the identified corners exactly
                corner_detected = corner_positions.copy()
            else:
                # Later iterations: occasionally try variations
                # Sample one of the non-corner holes and try it as a corner
                non_corner_mask = np.ones(n_detected, dtype=bool)
                non_corner_mask[corner_indices] = False
                if np.any(non_corner_mask):
                    # Replace one corner with a random non-corner hole
                    replace_idx = np.random.randint(4)
                    non_corner_indices = np.where(non_corner_mask)[0]
                    replacement = np.random.choice(non_corner_indices)

                    corner_detected = corner_positions.copy()
                    corner_detected[replace_idx] = hole_positions[replacement]
                else:
                    corner_detected = corner_positions.copy()

            # Compute homography from detected corners to known corners
            # Order: tl, tr, br, bl -> [0,0], [180,0], [180,101.25], [0,101.25]
            H, _ = cv2.findHomography(corner_detected, known_corners, 0)

            if H is None:
                continue

            # Transform ALL detected holes using this homography
            ones = np.ones((n_detected, 1))
            detected_homogeneous = np.hstack([hole_positions, ones])
            transformed = (H @ detected_homogeneous.T).T

            # Normalize homogeneous coordinates (avoid division by zero)
            w = transformed[:, 2:3]
            if np.any(np.abs(w) < 1e-8):
                continue  # Skip this bad homography
            transformed[:, :2] /= w

            # For each detected hole, find minimum distance to any known position
            # Also track WHICH known position it's closest to
            residuals = np.zeros(n_detected)
            closest_known = np.zeros(n_detected, dtype=int)

            for i in range(n_detected):
                distances = np.linalg.norm(known_positions - transformed[i, :2], axis=1)
                closest_known[i] = np.argmin(distances)
                residuals[i] = distances[closest_known[i]]

            # Better outlier detection: use bidirectional matching
            # For each known position, find the closest detected hole
            matched_detected_indices = set()
            for known_idx in range(len(known_positions)):
                # Find all detected holes that claim this known position as closest
                candidates = np.where(closest_known == known_idx)[0]
                if len(candidates) > 0:
                    # Among candidates, pick the one with smallest residual
                    best_candidate = candidates[np.argmin(residuals[candidates])]
                    matched_detected_indices.add(best_candidate)

            # Inliers are holes that got matched AND have low residual
            threshold = 25.0  # mm
            inliers = np.array([
                (i in matched_detected_indices) and (residuals[i] < threshold)
                for i in range(n_detected)
            ])
            inlier_count = np.sum(inliers)

            # Update best hypothesis
            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                best_inliers = inliers
                best_homography = H
                best_residuals = residuals

        if best_inliers is None or best_inlier_count < 4:
            raise ValueError(f"RANSAC failed: only found {best_inlier_count} inliers (need at least 4)")

        # Extract inlier holes and match them to known positions
        inlier_positions = hole_positions[best_inliers]
        inlier_residuals = best_residuals[best_inliers]

        # Transform inliers to known coordinate system
        ones = np.ones((len(inlier_positions), 1))
        inlier_homogeneous = np.hstack([inlier_positions, ones])
        transformed_inliers = (best_homography @ inlier_homogeneous.T).T

        # Normalize (with safety check)
        w = transformed_inliers[:, 2:3]
        transformed_inliers[:, :2] /= np.maximum(w, 1e-8)  # Avoid division by zero

        # Match each inlier to its closest known position
        # Use bidirectional matching to ensure 1-to-1 correspondence
        matched_detected = []
        matched_known = []
        used_known = set()

        # Build list of (inlier_idx, known_idx, distance) tuples
        matches = []
        for i, pos in enumerate(transformed_inliers[:, :2]):
            distances = np.linalg.norm(known_positions - pos, axis=1)
            closest_idx = np.argmin(distances)
            matches.append((i, closest_idx, distances[closest_idx]))

        # Sort by distance (best matches first)
        matches.sort(key=lambda x: x[2])

        # Assign matches, ensuring each known position used only once
        for inlier_idx, known_idx, dist in matches:
            if known_idx not in used_known:
                matched_detected.append(inlier_positions[inlier_idx])
                matched_known.append(known_positions[known_idx])
                used_known.add(known_idx)

        matched_detected = np.array(matched_detected, dtype=np.float32)
        matched_known = np.array(matched_known, dtype=np.float32)

        if debug:
            print(f"Final matching: {len(matched_detected)} correspondences")
            print(f"Matched known indices: {sorted(used_known)}")

        # Prepare outlier info for debugging
        outlier_info = {
            'total_detected': n_detected,
            'num_inliers': best_inlier_count,
            'num_outliers': n_detected - best_inlier_count,
            'inlier_mask': best_inliers,
            'residuals': best_residuals,
            'all_positions': hole_positions,
            'best_homography': best_homography  # IMPORTANT: Return the RANSAC homography
        }

        if debug:
            print(f"RANSAC: {best_inlier_count}/{n_detected} inliers, {outlier_info['num_outliers']} outliers rejected")
            print(f"Residuals - median: {np.median(best_residuals):.2f}mm, max inlier: {np.max(best_residuals[best_inliers]):.2f}mm")
            print(f"Inlier residuals: {sorted(best_residuals[best_inliers])[:5]} ... {sorted(best_residuals[best_inliers])[-3:]}")

            # Show which holes were rejected
            outlier_indices = np.where(~best_inliers)[0]
            if len(outlier_indices) > 0:
                print(f"Outlier positions (pixels): {hole_positions[outlier_indices].tolist()}")
                print(f"Outlier residuals: {best_residuals[outlier_indices].tolist()}")

        return matched_detected, matched_known, best_inliers, outlier_info
    
    def calibrate(self, image, debug=False):
        """
        Detect holes and calculate homography matrix with RANSAC outlier rejection

        Args:
            image: Input image
            debug: Show debug visualizations

        Returns:
            True if calibration successful, False otherwise
        """
        detected_holes = self.detect_holes(image, debug=debug)

        if len(detected_holes) < 4:
            print(f"Error: Only found {len(detected_holes)} holes, need at least 4")
            return False

        try:
            # Use RANSAC to match holes and reject outliers
            detected_points, known_points, inlier_mask, outlier_info = \
                self.match_holes_to_pattern(detected_holes, debug=debug)

            num_inliers = outlier_info['num_inliers']
            num_outliers = outlier_info['num_outliers']

            print(f"Matched {num_inliers} holes to pattern")

            if num_outliers > 0:
                print(f"  > Rejected {num_outliers} outlier(s) (likely from object, not frame)")

            # Warn if we lost too many holes
            if num_inliers < 10:
                print(f"  ! Warning: Only {num_inliers} holes matched. Calibration may be less accurate.")

            # Use the homography from RANSAC (don't recompute!)
            self.homography_matrix = outlier_info['best_homography']

            if self.homography_matrix is None:
                print("Error: Failed to calculate homography")
                return False

            if debug:
                print(f"Using RANSAC homography matrix")

            # Calculate approximate scale from homography
            # The homography maps from pixels to mm, so extract scale from it
            # For a similarity transform, scale ≈ sqrt(h00² + h01²)
            h00, h01 = self.homography_matrix[0, 0], self.homography_matrix[0, 1]
            h10, h11 = self.homography_matrix[1, 0], self.homography_matrix[1, 1]
            scale_x = np.sqrt(h00**2 + h10**2)
            scale_y = np.sqrt(h01**2 + h11**2)
            self.mm_per_pixel = (scale_x + scale_y) / 2.0

            print(f"Calibration successful! Scale: {self.mm_per_pixel:.4f} mm/pixel")

            # Store outlier info for visualization
            self.last_outlier_info = outlier_info
            self.last_image = image

            # Show debug visualization if requested
            if debug:
                self._visualize_hole_classification(image, outlier_info)

            return True

        except Exception as e:
            print(f"Error during calibration: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _visualize_hole_classification(self, image, outlier_info):
        """
        Visualize which holes were classified as inliers (frame) vs outliers (object)

        Args:
            image: Original input image
            outlier_info: Dictionary with inlier_mask, residuals, all_positions
        """
        debug_img = image.copy()

        all_positions = outlier_info['all_positions']
        inlier_mask = outlier_info['inlier_mask']
        residuals = outlier_info['residuals']

        # Draw all holes with color coding
        for i, (x, y) in enumerate(all_positions):
            is_inlier = inlier_mask[i]
            residual = residuals[i]

            if is_inlier:
                # Inliers in green (frame holes)
                color = (0, 255, 0)
                label = f"{residual:.1f}mm"
                cv2.circle(debug_img, (int(x), int(y)), 8, color, 2)
                cv2.putText(debug_img, label, (int(x) + 10, int(y) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            else:
                # Outliers in red (object holes)
                color = (0, 0, 255)
                label = "OUTLIER"
                cv2.circle(debug_img, (int(x), int(y)), 8, color, 2)
                cv2.putText(debug_img, label, (int(x) + 10, int(y) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Add summary text
        summary = f"Inliers: {outlier_info['num_inliers']} (green) | Outliers: {outlier_info['num_outliers']} (red)"
        cv2.putText(debug_img, summary, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(debug_img, summary, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

        cv2.imshow('Hole Classification (RANSAC)', debug_img)
        return debug_img
    
    def warp_to_calibrated_view(self, image):
        """
        Apply homography to warp image to calibrated top-down view

        Args:
            image: Input image

        Returns:
            Warped image in calibrated coordinate system
        """
        if self.homography_matrix is None:
            raise ValueError("Must calibrate before warping")

        # Output size based on frame dimensions (use a scale factor for resolution)
        scale = 10  # pixels per mm
        output_width = int(self.frame_width_mm * scale)
        output_height = int(self.frame_height_mm * scale)

        # The homography from RANSAC maps pixels to millimeters
        # We need to scale it to map pixels to output pixels (at 10 pixels/mm)
        scale_matrix = np.array([
            [scale, 0, 0],
            [0, scale, 0],
            [0, 0, 1]
        ], dtype=np.float64)

        # Compose: output_pixels = scale_matrix @ (homography @ input_pixels)
        adjusted_homography = scale_matrix @ self.homography_matrix

        warped = cv2.warpPerspective(image, adjusted_homography, (output_width, output_height))
        return warped


class ObjectMeasurer:
    """Measures objects in the calibrated frame"""
    
    def __init__(self, calibration_frame):
        """
        Initialize with a calibrated frame
        
        Args:
            calibration_frame: CalibrationFrame instance that has been calibrated
        """
        self.frame = calibration_frame
    
    def segment_object(self, warped_image, threshold=200, debug=False):
        """
        Segment object from white background
        
        Args:
            warped_image: Calibrated/warped image
            threshold: Threshold value for segmentation (lower = darker objects detected)
            debug: Show debug visualizations
            
        Returns:
            Binary mask of the object
        """
        gray = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
        
        # Invert threshold to find darker objects on white background
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
        
        # Remove small noise
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        if debug:
            cv2.imshow('Object Segmentation', binary)
        
        return binary
    
    def _score_contour(self, contour, threshold_value, image_shape):
        """
        Score a contour based on multiple quality metrics

        Returns a score between 0 and 1 (higher = better object candidate)
        """
        h, w = image_shape[:2]
        total_area = h * w

        # Get contour properties
        area = cv2.contourArea(contour)
        x, y, cw, ch = cv2.boundingRect(contour)
        bbox_area = cw * ch

        # Compute convex hull for solidity calculation
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)

        # 1. SIZE SCORE - Prefer objects that are 5-40% of frame area
        optimal_area_ratio = 0.15  # 15% is typical for a tool
        area_ratio = area / total_area
        size_score = np.exp(-((area_ratio - optimal_area_ratio)**2) / (2 * 0.1**2))

        # 2. SHAPE SCORE - Prefer solid, compact shapes
        solidity = area / hull_area if hull_area > 0 else 0
        extent = area / bbox_area if bbox_area > 0 else 0
        shape_score = (solidity * 0.6 + extent * 0.4)  # Solidity more important

        # 3. COMPLETENESS SCORE - Prefer shapes that fill their bounding box well
        aspect_ratio = max(cw, ch) / max(min(cw, ch), 1)
        # Penalize extreme aspect ratios (likely noise or partial detection)
        aspect_penalty = 1.0 if aspect_ratio < 15 else (15.0 / aspect_ratio)
        completeness_score = extent * aspect_penalty

        # 4. CONFIDENCE SCORE - Prefer mid-range thresholds (more reliable)
        # Threshold 210 is ideal, penalize extremes
        threshold_diff = abs(threshold_value - 210)
        confidence_score = np.exp(-(threshold_diff**2) / (2 * 40**2))

        # Weighted combination
        total_score = (
            0.35 * size_score +
            0.30 * shape_score +
            0.25 * completeness_score +
            0.10 * confidence_score
        )

        return total_score, {
            'size': size_score,
            'shape': shape_score,
            'completeness': completeness_score,
            'confidence': confidence_score,
            'area_ratio': area_ratio,
            'solidity': solidity,
            'extent': extent
        }

    def measure_object(self, warped_image, threshold=200, debug=False):
        """
        Measure the object in the calibrated image using adaptive multi-threshold segmentation

        Args:
            warped_image: Calibrated/warped image
            threshold: Base threshold (used as fallback, but multi-threshold is primary)
            debug: Show debug visualizations

        Returns:
            Dictionary with measurements
        """
        h, w = warped_image.shape[:2]
        total_area = h * w

        # Edge margin and size filters
        edge_margin = max(10, int(min(h, w) * 0.02))
        min_object_area = total_area * 0.001
        max_object_area = total_area * 0.6
        max_width = w * 0.95
        max_height = h * 0.95

        # ADAPTIVE MULTI-THRESHOLD SEGMENTATION
        # Try multiple thresholds to handle different object types
        thresholds_to_try = [170, 190, 210, 230, 245]

        all_candidates = []

        for thresh in thresholds_to_try:
            binary = self.segment_object(warped_image, threshold=thresh, debug=False)
            contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                x, y, cw, ch = cv2.boundingRect(contour)

                # Basic filters
                is_away_from_edges = (x > edge_margin and y > edge_margin and
                                     x + cw < w - edge_margin and y + ch < h - edge_margin)
                is_reasonable_size = (min_object_area < area < max_object_area)
                is_not_full_frame = (cw < max_width and ch < max_height)

                if is_away_from_edges and is_reasonable_size and is_not_full_frame:
                    # Score this contour
                    score, metrics = self._score_contour(contour, thresh, warped_image.shape)

                    all_candidates.append({
                        'contour': contour,
                        'threshold': thresh,
                        'score': score,
                        'metrics': metrics,
                        'area': area
                    })

        if len(all_candidates) == 0:
            return {"error": "No object detected at any threshold (tried 170-245)"}

        # Select the best-scoring contour
        best_candidate = max(all_candidates, key=lambda x: x['score'])
        object_contour = best_candidate['contour']

        if debug:
            print(f"\nAdaptive Segmentation Results:")
            print(f"  Tried {len(thresholds_to_try)} thresholds, found {len(all_candidates)} candidates")
            print(f"  Best: threshold={best_candidate['threshold']}, score={best_candidate['score']:.3f}")
            print(f"  Metrics: size={best_candidate['metrics']['size']:.2f}, "
                  f"shape={best_candidate['metrics']['shape']:.2f}, "
                  f"completeness={best_candidate['metrics']['completeness']:.2f}")
            print(f"  Solidity={best_candidate['metrics']['solidity']:.2f}, "
                  f"extent={best_candidate['metrics']['extent']:.2f}")

            # Show top 3 candidates
            top_3 = sorted(all_candidates, key=lambda x: x['score'], reverse=True)[:3]
            for i, cand in enumerate(top_3):
                print(f"  #{i+1}: thresh={cand['threshold']}, score={cand['score']:.3f}, "
                      f"area={cand['area']:.0f}px²")

        # Warn if best score is low
        if best_candidate['score'] < 0.3:
            print(f"  ! Warning: Low confidence detection (score={best_candidate['score']:.2f})")
            print(f"    Object may not be clearly visible or might need better lighting")
        
        # Calculate measurements in pixels
        perimeter_pixels = cv2.arcLength(object_contour, closed=True)
        area_pixels = cv2.contourArea(object_contour)
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(object_contour)
        
        # Convert to mm (warped image scale is 10 pixels per mm)
        scale = 10  # pixels per mm (from warp_to_calibrated_view)
        
        measurements = {
            "perimeter_mm": perimeter_pixels / scale,
            "area_mm2": area_pixels / (scale * scale),
            "bounding_box_width_mm": w / scale,
            "bounding_box_height_mm": h / scale,
            "contour": object_contour,
            "bounding_box": (x, y, w, h)
        }
        
        if debug:
            debug_img = warped_image.copy()
            cv2.drawContours(debug_img, [object_contour], -1, (0, 255, 0), 2)
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.imshow('Object Measurement', debug_img)
            
            print(f"\nMeasurements:")
            print(f"  Perimeter: {measurements['perimeter_mm']:.2f} mm")
            print(f"  Area: {measurements['area_mm2']:.2f} mm²")
            print(f"  Bounding box: {measurements['bounding_box_width_mm']:.2f} x {measurements['bounding_box_height_mm']:.2f} mm")
        
        return measurements


def main():
    parser = argparse.ArgumentParser(description='Measure objects using calibration frame')
    parser.add_argument('image', help='Path to image file')
    parser.add_argument('--debug', action='store_true', help='Show debug visualizations')
    parser.add_argument('--threshold', type=int, default=200, 
                       help='Segmentation threshold (default: 200, lower for darker objects)')
    parser.add_argument('--save-output', help='Save annotated output image to this path')
    
    args = parser.parse_args()
    
    # Load image
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Could not load image from {args.image}")
        return
    
    print(f"Loaded image: {image.shape[1]}x{image.shape[0]} pixels")
    
    # Create calibration frame (180mm x 101.25mm)
    frame = CalibrationFrame(frame_width_mm=180.0, frame_height_mm=101.25)
    
    # Calibrate
    print("\nCalibrating...")
    if not frame.calibrate(image, debug=args.debug):
        print("Calibration failed!")
        return
    
    # Warp image to calibrated view
    print("\nWarping to calibrated view...")
    warped = frame.warp_to_calibrated_view(image)
    
    if args.debug:
        cv2.imshow('Warped Calibrated View', warped)
    
    # Measure object
    print("\nMeasuring object...")
    measurer = ObjectMeasurer(frame)
    measurements = measurer.measure_object(warped, threshold=args.threshold, debug=args.debug)
    
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
    
    # Save output if requested
    if args.save_output and "contour" in measurements:
        output = warped.copy()
        cv2.drawContours(output, [measurements['contour']], -1, (0, 255, 0), 3)
        x, y, w, h = measurements['bounding_box']
        cv2.rectangle(output, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.imwrite(args.save_output, output)
        print(f"\nSaved annotated output to: {args.save_output}")
    
    if args.debug:
        print("\nPress any key to close windows...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
