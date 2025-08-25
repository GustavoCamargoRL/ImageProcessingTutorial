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

#original = cv2.imread('exemplo.png')

# Load image in grayscale
#gray = cv2.imread('exemplo.png', cv2.IMREAD_GRAYSCALE)
#show_image(gray, cmap='gray', title='Original Grayscale Image')

# Plot histogram
#plt.hist(gray.ravel(), 256, [0, 256])
#plt.title('Histogram')
#plt.xlabel('Pixel Intensity')
#plt.ylabel('Frequency')
#plt.show()

# Oceano
coral_live = cv2.imread('coral_v.png', cv2.IMREAD_GRAYSCALE)
show_image(coral_live, cmap='gray', title='Original Grayscale Image')
coral_dead = cv2.imread('coral_m.png', cv2.IMREAD_GRAYSCALE)
show_image(coral_dead, cmap='gray', title='Original Grayscale Image')

# Plot histogram
plt.hist(coral_live.ravel(), 256, [0, 256], label='Live Coral', color='g', alpha=0.5)
plt.hist(coral_dead.ravel(), 256, [0, 256], label='Dead Coral', color='r', alpha=0.5)
plt.title('Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.legend(labels=['Live Coral', 'Dead Coral'])
plt.show()

# Plot colour histogram
#color = ('b','g','r')
#for i,col in enumerate(color):
#    histr = cv2.calcHist([original],[i],None,[256],[0,256])
#    plt.plot(histr,color = col)
#    plt.xlim([0,256])
#plt.show()

# Apply histogram equalization
#equalized = cv2.equalizeHist(gray)
#show_image(equalized, cmap='gray', title='Equalized Image')

# Compare histograms
#plt.hist(equalized.ravel(), 256, [0, 256])
#plt.title('Equalized Histogram')
#plt.xlabel('Pixel Intensity')
#plt.ylabel('Frequency')
#plt.show()

# Equalize histogram on all 3 channels
#channels = cv2.split(original)
#eq_channels = [cv2.equalizeHist(ch) for ch in channels]
#equalized_color = cv2.merge(eq_channels)
#show_image(cv2.cvtColor(equalized_color, cv2.COLOR_BGR2RGB), title='Equalized Color Image')

# Plot colour histogram equalized
#color = ('b','g','r')
#for i,col in enumerate(color):
#    histr = cv2.calcHist([equalized_color],[i],None,[256],[0,256])
#    plt.plot(histr,color = col)
#    plt.xlim([0,256])
#plt.show()

# Add Gaussian noise to a colored image
#mean = 0
#stddev = 50
#gaussian_noise_color = np.random.normal(mean, stddev, original.shape).astype(np.float32)
#noisy_color_image = cv2.add(original.astype(np.float32), gaussian_noise_color)
#noisy_color_image = np.clip(noisy_color_image, 0, 255).astype(np.uint8)
#show_image(cv2.cvtColor(noisy_color_image, cv2.COLOR_BGR2RGB), title='Noisy Color Image')

#cv2.imwrite('noise_lenna.png', noisy_color_image)

# Mean filter
#blur_mean = cv2.blur(noisy_color_image, (10, 10))
#show_image(cv2.cvtColor(blur_mean, cv2.COLOR_BGR2RGB), cmap=None, title='Mean Blur')

# Gaussian filter
#blur_gaussian = cv2.GaussianBlur(noisy_color_image, (13, 13), 0)
#show_image(cv2.cvtColor(blur_gaussian, cv2.COLOR_BGR2RGB), cmap=None, title='Gaussian Blur')

# Median filter
#blur_median = cv2.medianBlur(noisy_color_image, 9)
#show_image(cv2.cvtColor(blur_median, cv2.COLOR_BGR2RGB), cmap=None, title='Median Blur')

# Bilateral filter
#bilateral = cv2.bilateralFilter(noisy_color_image,9,75,75)
#show_image(cv2.cvtColor(bilateral, cv2.COLOR_BGR2RGB), cmap=None, title='Sharpened Image opencv')

# Fixed threshold
_, binary_fixed = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
show_image(binary_fixed, cmap='gray', title='Fixed Threshold (127)')

# Adaptive threshold (Gaussian)
binary_adaptive = cv2.adaptiveThreshold(
    gray, 255, 
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    cv2.THRESH_BINARY, 
    11, 2
)
show_image(binary_adaptive, cmap='gray', title='Adaptive Threshold (Gaussian)')