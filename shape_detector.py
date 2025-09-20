import cv2
import numpy as np

def detect_shapes(image_path):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found. Check the path.")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect edges
    edged = cv2.Canny(blurred, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Approximate the contour
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)

        # Get bounding box for placing text
        x, y, w, h = cv2.boundingRect(approx)

        # Detect shape by number of vertices
        shape = "Unidentified"
        if len(approx) == 3:
            shape = "Triangle"
        elif len(approx) == 4:
            # Check aspect ratio to differentiate square vs rectangle
            aspect_ratio = float(w) / h
            if 0.85 <= aspect_ratio <= 1.15:
                shape = "Square"
            else:
                shape = "Rectangle"
        elif len(approx) > 6:
            shape = "Circle"
        else:
            shape = f"Polygon ({len(approx)} sides)"

        # Draw contour and shape name
        cv2.drawContours(image, [approx], -1, (0, 255, 0), 3)
        cv2.putText(image, shape, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 0, 0), 2)

    return image


if __name__ == "__main__":
    # Example usage
    input_path = "shapes.png"   # replace with your test image
    output = detect_shapes(input_path)

    cv2.imshow("Detected Shapes", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
