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

original = cv2.imread('exemplo.png')

# Load image in grayscale
gray = cv2.imread('exemplo.png', cv2.IMREAD_GRAYSCALE)
show_image(gray, cmap='gray', title='Original Grayscale Image')

# Plot histogram
plt.hist(gray.ravel(), 256, [0, 256])
plt.title('Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.show()

# Plot colour histogram
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([original],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.show()

# Apply histogram equalization
equalized = cv2.equalizeHist(gray)
show_image(equalized, cmap='gray', title='Equalized Image')

# Compare histograms
plt.hist(equalized.ravel(), 256, [0, 256])
plt.title('Equalized Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.show()

# Equalize histogram on all 3 channels
channels = cv2.split(original)
eq_channels = [cv2.equalizeHist(ch) for ch in channels]
equalized_color = cv2.merge(eq_channels)
show_image(cv2.cvtColor(equalized_color, cv2.COLOR_BGR2RGB), title='Equalized Color Image')

# Plot colour histogram equalized
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([equalized_color],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.show()