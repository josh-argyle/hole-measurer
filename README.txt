CALIBRATION FRAME OBJECT MEASUREMENT SYSTEM
==========================================

This system measures 2D objects placed on a white background within a calibration
frame with fiducial markers (white holes in a black frame).


FRAME SPECIFICATIONS
--------------------
- Frame size: 180mm (width) x 101.25mm (height)
- Black frame with white circular holes as fiducial markers
- 15 total holes:
  * 4 corners
  * Top edge: 6 holes at 1/2, 1/4, 1/8, 1/16, 1/32, 1/64 positions from left
  * Left edge: 5 holes at 1/2, 1/4, 1/8, 1/16, 1/32 positions from top


REQUIREMENTS
------------
- Python 3.7+
- OpenCV (cv2): pip install opencv-python
- NumPy: pip install numpy

Installation:
  pip install opencv-python numpy


HOW IT WORKS
------------
1. Detects the white circular holes in the frame
2. Matches detected holes to the known pattern
3. Calculates a homography (perspective transformation) to correct camera angle
4. Warps the image to a top-down calibrated view
5. Segments objects from the white background
6. Measures perimeter, area, and bounding box dimensions


USAGE
-----

Basic usage:
  python measure_object.py image.jpg

With debug visualization:
  python measure_object.py image.jpg --debug

Adjust threshold for darker/lighter objects:
  python measure_object.py image.jpg --threshold 180

Save annotated output:
  python measure_object.py image.jpg --save-output result.jpg

Full options:
  python measure_object.py --help


COMMAND LINE OPTIONS
--------------------
image               Path to the image file (required)
--debug            Show debug visualizations at each step
--threshold N      Segmentation threshold (default: 200)
                   - Lower values (150-180) for darker objects
                   - Higher values (220-240) for very light objects
--save-output FILE Save annotated image showing detected contour


TIPS FOR BEST RESULTS
----------------------

Camera Setup:
- Position camera directly above the frame (perpendicular)
- Ensure good, even lighting with minimal shadows
- Avoid reflections on shiny objects
- Keep consistent distance between camera and frame

Image Quality:
- Higher resolution = better accuracy
- Good contrast between object and white background
- Make sure all holes are clearly visible
- Avoid motion blur

Object Placement:
- Place object fully within the white area
- Ensure object doesn't cover the holes
- Use contrasting objects (dark on white works best)
- Flat objects work best (2D measurement)


TROUBLESHOOTING
---------------

"Only found X holes, need at least 4"
  → Improve lighting, ensure holes are bright white
  → Check that camera can see all corners clearly
  → Try increasing image resolution

"No object detected"
  → Adjust --threshold parameter
  → Ensure object contrasts with white background
  → Object might be too light (try --threshold 180)
  → Check object is fully in frame

Poor accuracy:
  → Ensure camera is perpendicular to frame
  → Check for lens distortion (use better camera if needed)
  → Verify frame is flat and not warped
  → More holes detected = better accuracy

Object not fully detected:
  → Adjust --threshold (lower for darker, higher for lighter)
  → Ensure even lighting without shadows
  → Check object isn't too close to hole markers


MEASUREMENTS EXPLAINED
----------------------

Perimeter (mm):
  The total length around the outer edge of the object.
  Useful for things like: length of wire, edge trim needed, etc.

Area (mm²):
  The total surface area of the object.
  Useful for: material coverage, size comparisons, etc.

Bounding Box (mm x mm):
  The smallest rectangle that contains the entire object.
  Gives you the maximum width and height dimensions.


EXAMPLE WORKFLOW
----------------

1. Print or 3D print the calibration frame
   - Must match exact dimensions: 180mm x 101.25mm
   - Black frame with white holes

2. Set up photography station
   - Mount camera directly above frame
   - Ensure even lighting
   - White surface inside frame

3. Take photo of frame with object
   - Place object (e.g., spanner, key, part) in center
   - Ensure all corner holes are visible
   - Take photo

4. Run measurement:
   python measure_object.py photo.jpg --debug

5. Review results:
   - Check detected holes (green dots)
   - Verify object outline (green contour)
   - Read measurements in console


TECHNICAL DETAILS
-----------------

Coordinate System:
  - Origin (0,0) at top-left corner of frame
  - X-axis increases to the right
  - Y-axis increases downward
  - All measurements in millimeters

Accuracy:
  - Typical accuracy: ±0.5mm with good setup
  - Improves with: more holes detected, higher resolution, better lighting
  - Limited by: camera resolution, lens distortion, setup alignment

Algorithm:
  1. Blob detection finds white circular holes
  2. Corner detection identifies frame corners
  3. Pattern matching assigns holes to known positions
  4. Homography calculation (cv2.findHomography with RANSAC)
  5. Image warping to normalized coordinate system
  6. Thresholding to segment object
  7. Contour detection and measurement
  8. Scale conversion using known frame dimensions


CODE STRUCTURE
--------------

CalibrationFrame class:
  - Stores frame dimensions and hole positions
  - Detects holes in images
  - Calculates homography matrix
  - Warps images to calibrated view

ObjectMeasurer class:
  - Segments objects from background
  - Measures perimeter, area, bounding box
  - Returns measurements in millimeters


EXTENDING THE CODE
------------------

To use in your own scripts:

  from measure_object import CalibrationFrame, ObjectMeasurer
  import cv2
  
  # Load image
  image = cv2.imread('photo.jpg')
  
  # Calibrate
  frame = CalibrationFrame(180.0, 101.25)
  frame.calibrate(image)
  
  # Warp and measure
  warped = frame.warp_to_calibrated_view(image)
  measurer = ObjectMeasurer(frame)
  results = measurer.measure_object(warped)
  
  print(f"Perimeter: {results['perimeter_mm']:.2f} mm")
  print(f"Area: {results['area_mm2']:.2f} mm²")


FUTURE IMPROVEMENTS
-------------------
- Support for multiple objects in one image
- More sophisticated shape analysis (convex hull, circularity, etc.)
- Lens distortion correction for wide-angle cameras
- Batch processing of multiple images
- Export measurements to CSV/JSON
- Real-time video measurement


SUPPORT & ISSUES
----------------
- Ensure frame dimensions match your actual printed/3D printed frame
- If holes are not detected, check image quality and lighting
- For very high precision, consider camera calibration to correct lens distortion
- Test with known-size objects first to verify accuracy


VERSION HISTORY
---------------
v1.0 - Initial release
  - Hole detection and pattern matching
  - Homography calculation
  - Object segmentation and measurement
  - Command line interface with debug mode
