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
#cleaned = cv2.erode(binary, kernel, iterations=2)
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