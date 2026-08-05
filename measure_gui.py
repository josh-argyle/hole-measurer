#!/usr/bin/env python3
"""
GUI for Hole Measurer
Simple interface for selecting images and measuring objects on calibration frame
"""

import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
from pathlib import Path
from measure_object import (CalibrationFrame, ObjectMeasurer, OUTPUT_DIR,
                            DEFAULT_INNER_WIDTH_MM, DEFAULT_INNER_HEIGHT_MM)


class MeasurementGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Hole Measurer - Object Measurement Tool")
        self.root.geometry("1200x800")

        # Inner-window dimensions of the calibration frame (mm)
        self.frame_width_mm = DEFAULT_INNER_WIDTH_MM
        self.frame_height_mm = DEFAULT_INNER_HEIGHT_MM

        # Current image path
        self.current_image_path = None
        self.result_image = None

        # Convex hull mode
        self.use_convex_hull = tk.BooleanVar(value=False)

        # Setup UI
        self.setup_ui()

    def setup_ui(self):
        """Create the GUI layout"""
        # Header
        header_frame = tk.Frame(self.root, bg="#2c3e50", height=60)
        header_frame.pack(fill=tk.X, side=tk.TOP)

        title_label = tk.Label(
            header_frame,
            text="Object Measurement Tool",
            font=("Arial", 20, "bold"),
            bg="#2c3e50",
            fg="white"
        )
        title_label.pack(pady=15)

        # Control panel
        control_frame = tk.Frame(self.root, bg="#ecf0f1", height=80)
        control_frame.pack(fill=tk.X, side=tk.TOP, padx=10, pady=10)

        # Select image button
        self.select_btn = tk.Button(
            control_frame,
            text="📁 Select Image",
            command=self.select_image,
            font=("Arial", 12, "bold"),
            bg="#3498db",
            fg="white",
            padx=20,
            pady=10,
            cursor="hand2"
        )
        self.select_btn.pack(side=tk.LEFT, padx=10)

        # Status label
        self.status_label = tk.Label(
            control_frame,
            text="No image selected",
            font=("Arial", 11),
            bg="#ecf0f1",
            fg="#7f8c8d"
        )
        self.status_label.pack(side=tk.LEFT, padx=20)

        # Convex hull checkbox
        self.convex_hull_check = tk.Checkbutton(
            control_frame,
            text="Convex Hull (for hollow objects)",
            variable=self.use_convex_hull,
            font=("Arial", 10),
            bg="#ecf0f1",
            activebackground="#ecf0f1"
        )
        self.convex_hull_check.pack(side=tk.LEFT, padx=10)

        # Progress bar
        self.progress = ttk.Progressbar(
            control_frame,
            mode='indeterminate',
            length=200
        )
        # Don't pack initially - only show when processing

        # Main content area
        content_frame = tk.Frame(self.root, bg="white")
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Image display area
        image_frame = tk.Frame(content_frame, bg="white", relief=tk.RIDGE, borderwidth=2)
        image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        image_label = tk.Label(image_frame, text="Result Image", font=("Arial", 12, "bold"), bg="white")
        image_label.pack(pady=5)

        # Canvas for image
        self.canvas = tk.Canvas(image_frame, bg="#f8f9fa")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Measurements panel
        measurements_frame = tk.Frame(content_frame, bg="#ecf0f1", width=300, relief=tk.RIDGE, borderwidth=2)
        measurements_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5)
        measurements_frame.pack_propagate(False)

        measurements_title = tk.Label(
            measurements_frame,
            text="Measurements",
            font=("Arial", 14, "bold"),
            bg="#ecf0f1"
        )
        measurements_title.pack(pady=10)

        # Measurement display
        self.measurement_text = tk.Text(
            measurements_frame,
            font=("Courier", 11),
            bg="white",
            relief=tk.FLAT,
            padx=15,
            pady=15,
            state=tk.DISABLED
        )
        self.measurement_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Footer
        footer_frame = tk.Frame(self.root, bg="#34495e", height=30)
        footer_frame.pack(fill=tk.X, side=tk.BOTTOM)

        footer_label = tk.Label(
            footer_frame,
            text=f"Powered by OpenCV • Frame inner window: {DEFAULT_INNER_WIDTH_MM} × {DEFAULT_INNER_HEIGHT_MM} mm",
            font=("Arial", 9),
            bg="#34495e",
            fg="white"
        )
        footer_label.pack(pady=5)

    def select_image(self):
        """Open file dialog to select an image"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp"),
                ("All files", "*.*")
            ]
        )

        if file_path:
            self.current_image_path = file_path
            self.status_label.config(text=f"Processing: {file_path.split('/')[-1]}")
            self.progress.pack(side=tk.LEFT, padx=10)
            self.progress.start()

            # Process image after a short delay (allows UI to update)
            self.root.after(100, self.process_image)

    def process_image(self):
        """Process the selected image and display results"""
        try:
            # Load image
            image = cv2.imread(self.current_image_path)
            if image is None:
                raise ValueError("Failed to load image")

            # Create calibration frame
            frame = CalibrationFrame(self.frame_width_mm, self.frame_height_mm)

            # Calibrate
            success = frame.calibrate(image, debug=False)
            if not success:
                raise ValueError("Calibration failed - ensure the full black frame is visible")

            # Warp to calibrated view
            warped = frame.warp_to_calibrated_view(image)

            # Measure object
            measurer = ObjectMeasurer(frame)
            results = measurer.measure_object(warped, debug=False, use_convex_hull=self.use_convex_hull.get())

            if "error" in results:
                raise ValueError(results["error"])

            # Draw results on warped image
            result_image = self.draw_results(warped, results)

            # Save annotated result to the output folder
            OUTPUT_DIR.mkdir(exist_ok=True)
            out_path = OUTPUT_DIR / f"{Path(self.current_image_path).stem}_measured.png"
            cv2.imwrite(str(out_path), result_image)

            # Display
            self.display_image(result_image)
            self.display_measurements(results)

            self.status_label.config(text=f"✓ Saved to {out_path.name} in output folder", fg="#27ae60")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to process image:\n{str(e)}")
            self.status_label.config(text=f"✗ Error processing image", fg="#e74c3c")

        finally:
            self.progress.stop()
            self.progress.pack_forget()

    def draw_results(self, warped_image, results):
        """Draw bounding box and measurements on the image"""
        result = warped_image.copy()

        # Get bounding box
        x, y, w, h = results['bounding_box']

        # Draw bounding box
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 3)

        # Draw contour
        cv2.drawContours(result, [results['contour']], -1, (0, 255, 0), 2)

        # Prepare dimension text
        width_mm = results['bounding_box_width_mm']
        height_mm = results['bounding_box_height_mm']

        # Draw width dimension (top)
        mid_x = x + w // 2
        cv2.line(result, (x, y - 20), (x + w, y - 20), (255, 0, 0), 2)
        cv2.line(result, (x, y - 25), (x, y - 15), (255, 0, 0), 2)  # Left tick
        cv2.line(result, (x + w, y - 25), (x + w, y - 15), (255, 0, 0), 2)  # Right tick

        width_text = f"{width_mm:.1f} mm"
        text_size = cv2.getTextSize(width_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(result,
                     (mid_x - text_size[0]//2 - 5, y - 50),
                     (mid_x + text_size[0]//2 + 5, y - 20),
                     (255, 255, 255), -1)
        cv2.putText(result, width_text,
                   (mid_x - text_size[0]//2, y - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Draw height dimension (right side)
        mid_y = y + h // 2
        cv2.line(result, (x + w + 20, y), (x + w + 20, y + h), (255, 0, 0), 2)
        cv2.line(result, (x + w + 15, y), (x + w + 25, y), (255, 0, 0), 2)  # Top tick
        cv2.line(result, (x + w + 15, y + h), (x + w + 25, y + h), (255, 0, 0), 2)  # Bottom tick

        height_text = f"{height_mm:.1f} mm"
        text_size = cv2.getTextSize(height_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]

        # Rotate text for vertical dimension
        cv2.rectangle(result,
                     (x + w + 30, mid_y - text_size[0]//2 - 5),
                     (x + w + 60, mid_y + text_size[0]//2 + 5),
                     (255, 255, 255), -1)
        cv2.putText(result, height_text,
                   (x + w + 35, mid_y + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        return result

    def display_image(self, cv_image):
        """Display OpenCV image on canvas"""
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_image)

        # Get canvas size
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        # If canvas hasn't been rendered yet, use default size
        if canvas_width <= 1:
            canvas_width = 800
            canvas_height = 600

        # Resize to fit canvas while maintaining aspect ratio
        img_width, img_height = pil_image.size
        scale = min(canvas_width / img_width, canvas_height / img_height) * 0.95

        new_width = int(img_width * scale)
        new_height = int(img_height * scale)

        pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(pil_image)

        # Display on canvas
        self.canvas.delete("all")
        self.canvas.create_image(canvas_width//2, canvas_height//2, image=photo, anchor=tk.CENTER)

        # Keep reference to prevent garbage collection
        self.canvas.image = photo
        self.result_image = cv_image

    def display_measurements(self, results):
        """Display measurements in the text widget"""
        self.measurement_text.config(state=tk.NORMAL)
        self.measurement_text.delete(1.0, tk.END)

        # Format measurements
        text = f"""
╔═══════════════════════════╗
║   MEASUREMENT RESULTS     ║
╚═══════════════════════════╝

Bounding Box:
  Width:  {results['bounding_box_width_mm']:>8.2f} mm
  Height: {results['bounding_box_height_mm']:>8.2f} mm

Perimeter: {results['perimeter_mm']:>8.2f} mm

Area:      {results['area_mm2']:>8.2f} mm²

═══════════════════════════

Note: Measurements are based
on frame inner window
({self.frame_width_mm} × {self.frame_height_mm} mm)
"""

        self.measurement_text.insert(1.0, text)
        self.measurement_text.config(state=tk.DISABLED)


def main():
    root = tk.Tk()
    app = MeasurementGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
