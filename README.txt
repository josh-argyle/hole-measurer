CALIBRATION FRAME OBJECT MEASUREMENT SYSTEM
==========================================

Measures 2D objects placed on a white background inside a plain black
rectangular frame. No fiducial holes are needed - calibration comes from
the inner edge of the frame itself.


FRAME SPECIFICATIONS
--------------------
- A flat black rectangular frame on/around a white background
- Default frame: 170 x 170 mm outer, 10 mm border
  -> inner opening 150 x 150 mm (this is the calibrated dimension)
- Small corner fillets (e.g. 1 mm radius) are fine - the software fits
  straight lines to the edges and intersects them, so rounded corners
  do not affect accuracy
- Border should be ~10 mm wide so it reads as solid black in the photo
- If your frame differs, measure the INNER opening (edge to edge) and
  pass --width and --height


REQUIREMENTS
------------
- Python 3.7+
- pip install opencv-python numpy


HOW IT WORKS
------------
1. Thresholds the image and finds the bright window enclosed by the
   black frame (regions touching the image border are ignored, so a
   white table around the frame is fine)
2. Fits straight lines to the window's 4 inner edges and intersects
   them for sub-pixel corner positions
3. Computes a homography from those corners to the known inner
   dimensions - this fully corrects camera angle/perspective
4. Warps to a top-down view at 10 pixels per mm; everything outside
   the window is cropped away
5. Segments the object from the white background (adaptive
   multi-threshold) and measures it


USAGE
-----
Basic:
  python measure_object.py image.jpg

Custom frame size (inner opening in mm):
  python measure_object.py image.jpg --width 150 --height 150

Other options:
  --debug             Show debug visualizations
  --threshold N       Segmentation fallback threshold (default 200)
  --convex-hull       For hollow objects (pliers, scissors)
  --save-output FILE  Save annotated image

GUI:
  python measure_gui.py


TIPS FOR BEST RESULTS
---------------------
- The entire black frame must be in the photo, not touching the image edge
- Even lighting; avoid strong shadows (shadows can be measured as object)
- Camera angle is corrected automatically, but a roughly top-down shot
  gives the best segmentation resolution
- Object must sit fully inside the window and contrast with the white
  background
- For sub-0.5 mm accuracy, avoid wide-angle lenses (lens distortion is
  not corrected)


MEASUREMENTS
------------
- Perimeter (mm): length around the object outline
- Area (mm^2): enclosed area
- Bounding box (mm x mm): smallest upright rectangle containing the object

Accuracy: calibration is typically well under 0.5 mm; overall accuracy is
limited by segmentation (lighting/shadows) and lens distortion.


EXTENDING THE CODE
------------------
  from measure_object import CalibrationFrame, ObjectMeasurer
  import cv2

  image = cv2.imread('photo.jpg')
  frame = CalibrationFrame(inner_width_mm=150.0, inner_height_mm=150.0)
  frame.calibrate(image)
  warped = frame.warp_to_calibrated_view(image)
  results = ObjectMeasurer(frame).measure_object(warped)
  print(results['bounding_box_width_mm'], results['bounding_box_height_mm'])


VERSION HISTORY
---------------
v2.0 - Frame-window calibration
  - Calibrates from the black frame's inner rectangle (no holes needed)
  - Edge-line fitting: immune to rounded corners, sub-pixel accuracy
  - Fixed 2-3% systematic scale error of the hole-based version
v1.0 - Initial release (hole-based fiducials)
