import cv2
import numpy as np
from matplotlib import pyplot as plt

# Helper function to display image
def show_image(img, cmap=None, title=''):
    plt.figure(figsize=(6, 6))
    plt.axis('off')
    if title:
        plt.title(title)
    plt.imshow(img, cmap=cmap)
    plt.show()

# Load image in grayscale
img = cv2.imread('exemplo.png')  # Replace with a suitable image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
show_image(gray, cmap='gray', title='Grayscale Image')

# Apply Gaussian blur to reduce noise
#blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#show_image(blurred, cmap='gray', title='Blurred Image')

# Sobel X and Y
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

# Convert back to uint8
#sobel_x = cv2.convertScaleAbs(sobel_x)
#sobel_y = cv2.convertScaleAbs(sobel_y)

# Display Sobel results
#show_image(sobel_x, cmap='gray', title='Sobel X')
#show_image(sobel_y, cmap='gray', title='Sobel Y')

# Laplacian
#laplacian = cv2.Laplacian(gray, cv2.CV_64F)
#laplacian = cv2.convertScaleAbs(laplacian)
#show_image(laplacian, cmap='gray', title='Laplacian')

# Apply Canny edge detector
edges = cv2.Canny(gray, 100, 200)
show_image(edges, cmap='gray', title='Canny Edges')

# Find contours from Canny edges
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on a copy of original image
img_contours = img.copy()
cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 2)
show_image(cv2.cvtColor(img_contours, cv2.COLOR_BGR2RGB), title='Contours')