import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread("coins.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur to reduce noise
blurred = cv2.GaussianBlur(gray, (7, 7), 0)

# Threshold the image (binary)
_, binary = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV)

# Define a kernel
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

# Apply morphological operations
cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=3)
#cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)

# Find contours
contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on the original image
result = image.copy()
cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

# Count objects
num_coins = len(contours)

# Show results
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.title("Binary Image")
plt.imshow(binary, cmap="gray")

plt.subplot(1, 3, 2)
plt.title("After Morphology")
plt.imshow(cleaned, cmap="gray")

plt.subplot(1, 3, 3)
plt.title(f"Detected Coins: {num_coins}")
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

plt.show()


def interactive_hsv_mask(image_path):
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    def nothing(x):
        pass

    # Create a window
    cv2.namedWindow('Mask Tuning', cv2.WINDOW_NORMAL)

    # Create trackbars for upper HSV values
    cv2.createTrackbar('H', 'Mask Tuning', 180, 180, nothing)
    cv2.createTrackbar('S', 'Mask Tuning', 255, 255, nothing)
    cv2.createTrackbar('V', 'Mask Tuning', 60, 255, nothing)

    while True:
        # Get current positions of trackbars
        h = cv2.getTrackbarPos('H', 'Mask Tuning')
        s = cv2.getTrackbarPos('S', 'Mask Tuning')
        v = cv2.getTrackbarPos('V', 'Mask Tuning')

        lower_black = np.array([0, 0, 0])
        upper_black = np.array([h, s, v])

        mask = cv2.inRange(hsv, lower_black, upper_black)

        # Show mask and original side by side
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        combined = np.hstack((image, mask_bgr))
        cv2.imshow('Mask Tuning', combined)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # ESC to exit
            break

    cv2.destroyAllWindows()
    print(f"Final upper_black: {upper_black}")

# Load the image
image = cv2.imread("oil.jpg")
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
interactive_hsv_mask("oil4.jpg")

# Define range for dark (oil spill) regions
# Lower values: near black, Upper values: dark gray
lower_black = np.array([0, 0, 0])
upper_black = np.array([120,255, 60])  # allow some dark gray

# Threshold to create mask
mask = cv2.inRange(hsv, lower_black, upper_black)

# Morphological operations to clean small noise
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)

# Find contours of oil spill regions
contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on the original image
result = image.copy()
cv2.drawContours(result, contours, -1, (0, 0, 255), 20)

# Show results
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

plt.subplot(1, 3, 2)
plt.title("Oil Spill Mask")
plt.imshow(mask_cleaned, cmap="gray")

plt.subplot(1, 3, 3)
plt.title("Detected Oil Spill")
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

plt.show()

print(f"Detected Oil Spill Regions: {len(contours)}")

# Get coordinates of white pixels
points = np.column_stack(np.where(mask_cleaned > 0))

# Apply PCA to find dominant direction
mean, eigenvectors = cv2.PCACompute(points.astype(np.float32), mean=None)
scale = 400  # length of the arrow
# Get center and principal direction
center = mean[0]  # (y, x) format
direction = eigenvectors[0]

# Make sure to swap order: (col, row) -> (x, y)
pt1 = (int(center[1]), int(center[0]))
pt2 = (int(center[1] + scale * direction[1]),
       int(center[0] + scale * direction[0]))
print(pt1, pt2  )

result = image.copy()
cv2.arrowedLine(result, pt1, pt2, (0, 0, 255), 20, tipLength=0.5)

plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title("Corrected Oil Spill Flow Direction")
plt.show()
