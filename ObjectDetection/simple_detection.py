from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse
import random as rng

rng.seed(12345)

# Global variables for trackbars
src = None
src_gray = None
min_area = 100
max_area = 50000
canny_thresh = 100
canny_ratio = 2
kernel_size = 3


def detect_objects():
    """Main function to detect objects and draw bounding boxes"""
    global src, src_gray, min_area, max_area, canny_thresh, canny_ratio, kernel_size

    if src is None:
        return

    # Convert to grayscale if not already
    if len(src.shape) == 3:
        gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    else:
        gray = src_gray.copy()

    # Apply Gaussian blur to reduce noise
    blurred = cv.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    # Apply Canny edge detection
    edges = cv.Canny(blurred, canny_thresh, canny_thresh * canny_ratio)

    # Apply morphological operations to close gaps
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Filter contours by area and create bounding boxes
    filtered_contours = []
    bounding_boxes = []

    for contour in contours:
        area = cv.contourArea(contour)
        if min_area < area < max_area:
            # Get bounding rectangle
            x, y, w, h = cv.boundingRect(contour)

            # Additional filtering: aspect ratio and size
            aspect_ratio = w / h
            if 0.2 < aspect_ratio < 5.0:  # Reasonable aspect ratio
                filtered_contours.append(contour)
                bounding_boxes.append((x, y, w, h))

    # Create output image
    output = src.copy()

    # Draw bounding boxes and contours
    for i, (contour, (x, y, w, h)) in enumerate(zip(filtered_contours, bounding_boxes)):
        # Generate a consistent color for each object
        color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))

        # Draw contour
        cv.drawContours(output, [contour], -1, color, 2)

        # Draw bounding rectangle
        cv.rectangle(output, (x, y), (x + w, y + h), color, 2)

        # Draw object number
        cv.putText(
            output, f"Object {i+1}", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )

        # Draw area information
        area = cv.contourArea(contour)
        cv.putText(
            output,
            f"Area: {int(area)}",
            (x, y + h + 20),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )

    # Display results
    cv.imshow("Object Detection", output)
    cv.imshow("Edges", edges)

    # Print detection summary
    print(f"Detected {len(filtered_contours)} objects")


def on_min_area_change(val):
    global min_area
    min_area = val
    detect_objects()


def on_max_area_change(val):
    global max_area
    max_area = val
    detect_objects()


def on_canny_thresh_change(val):
    global canny_thresh
    canny_thresh = val
    detect_objects()


def on_kernel_size_change(val):
    global kernel_size
    kernel_size = max(3, val | 1)  # Ensure odd number >= 3
    detect_objects()


def main():
    global src, src_gray

    parser = argparse.ArgumentParser(description="Object Detection with Bounding Boxes")
    parser.add_argument("--input", help="Path to input image.", default="stuff.jpg")
    parser.add_argument(
        "--camera", action="store_true", help="Use camera instead of image file"
    )
    args = parser.parse_args()

    # Load image or use camera
    if args.camera:
        cap = cv.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return

        print("Press 'q' to quit, 's' to save current frame")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            src = frame
            src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
            detect_objects()

            key = cv.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                cv.imwrite("detected_objects.jpg", src)
                print("Frame saved as 'detected_objects.jpg'")

        cap.release()
    else:
        # Load image from file
        src = cv.imread(cv.samples.findFile(args.input))
        if src is None:
            print("Could not open or find the image:", args.input)
            print("Trying to use camera instead...")
            cap = cv.VideoCapture(0)
            if not cap.isOpened():
                print("Error: Could not open camera")
                return
            ret, src = cap.read()
            cap.release()
            if not ret:
                print("Error: Could not capture from camera")
                return

        # Convert to grayscale
        src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

        # Create windows and trackbars
        cv.namedWindow("Object Detection", cv.WINDOW_AUTOSIZE)
        cv.namedWindow("Edges", cv.WINDOW_AUTOSIZE)

        # Create trackbars for parameter adjustment
        cv.createTrackbar(
            "Min Area", "Object Detection", min_area, 10000, on_min_area_change
        )
        cv.createTrackbar(
            "Max Area", "Object Detection", max_area, 100000, on_max_area_change
        )
        cv.createTrackbar(
            "Canny Threshold",
            "Object Detection",
            canny_thresh,
            255,
            on_canny_thresh_change,
        )
        cv.createTrackbar(
            "Blur Kernel Size",
            "Object Detection",
            kernel_size,
            15,
            on_kernel_size_change,
        )

        # Initial detection
        detect_objects()

        print("Controls:")
        print("- Adjust trackbars to fine-tune detection")
        print("- Press 'q' to quit")
        print("- Press 's' to save result")

        while True:
            key = cv.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                cv.imwrite("detected_objects.jpg", src)
                print("Result saved as 'detected_objects.jpg'")

    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
