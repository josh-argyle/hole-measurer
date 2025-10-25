# Hole Measurer GUI

A simple graphical interface for measuring objects using the calibration frame.

## Features

- 📁 **Easy file selection** - Click to browse and select images
- 📏 **Visual measurements** - See bounding box with dimensions overlaid
- 📊 **Detailed results** - View perimeter, area, and bounding box measurements
- 🎯 **Automatic processing** - RANSAC outlier rejection and adaptive segmentation
- ✨ **Clean interface** - Modern, intuitive design

## Usage

1. **Launch the GUI:**
   ```bash
   python measure_gui.py
   ```

2. **Select an image:**
   - Click "📁 Select Image" button
   - Choose an image with your calibration frame and object

3. **View results:**
   - The measured object appears with a green contour
   - Blue bounding box shows the object's dimensions
   - Red dimension lines display width and height in millimeters
   - Measurements panel shows detailed metrics

## Requirements

- Python 3.7+
- opencv-python
- numpy
- Pillow

Install with:
```bash
pip install -r requirements.txt
```

## How It Works

The GUI uses the same robust measurement system as the command-line tool:

1. **Calibration**: Detects 15 holes in the calibration frame using RANSAC
2. **Outlier Rejection**: Filters out spurious detections (e.g., tool openings)
3. **Perspective Correction**: Warps image to top-down view
4. **Adaptive Segmentation**: Uses multiple thresholds (170-245) to detect objects
5. **Measurement**: Calculates accurate dimensions in millimeters

## Screenshot

The GUI displays:
- **Header**: Application title
- **Control Panel**: File selection and status
- **Main View**: Result image with annotations
- **Measurements Panel**: Detailed numeric results
- **Footer**: Information about the system

## Tips

- Ensure all 15 calibration holes are visible in the image
- Place objects on the white area inside the black frame
- Use good, even lighting for best results
- The system automatically handles chrome, steel, and plastic objects

## Troubleshooting

**"Calibration failed"**
- Check that all frame holes are visible
- Ensure good lighting and focus
- Make sure the image includes the complete frame

**"No object detected"**
- Verify the object contrasts with the white background
- Try adjusting camera angle or lighting
- Ensure the object is inside the frame (not touching edges)

## Technical Details

- **Frame dimensions**: 180mm × 101.25mm
- **Resolution**: 10 pixels per millimeter after warping
- **Accuracy**: Sub-millimeter precision on calibrated frame
- **Processing time**: Typically 1-2 seconds per image
