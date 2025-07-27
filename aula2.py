import cv2
import numpy as np
from matplotlib import pyplot as plt

# Helper function to show image
def show_image(img, cmap=None, title=''):
    plt.figure(figsize=(6, 6))
    plt.axis('off')
    if title:
        plt.title(title)
    plt.imshow(img, cmap=cmap)
    plt.show()

# Load image in grayscale
gray = cv2.imread('exemplo.png', cv2.IMREAD_GRAYSCALE)
show_image(gray, cmap='gray', title='Original Grayscale Image')

# Plot histogram
plt.hist(gray.ravel(), 256, [0, 256])
plt.title('Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.show()