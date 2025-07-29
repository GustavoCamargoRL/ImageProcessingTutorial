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
img = cv2.imread('shapes.png')  # Replace with a suitable image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
show_image(gray, cmap='gray', title='Grayscale Image')

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
show_image(blurred, cmap='gray', title='Blurred Image')