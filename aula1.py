import cv2
import numpy as np
from matplotlib import pyplot as plt

# Helper function to display images using matplotlib
def show_image(img, cmap=None):
    plt.axis('off')
    plt.imshow(img, cmap=cmap)
    plt.show()

# 2. Load an image from file
image = cv2.imread('exemplo.png')  # Replace with a valid image path
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#show_image(image_rgb)

# 3. Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#show_image(gray, cmap='gray')

# 4. Resize and crop the image
resized = cv2.resize(image, (200, 200))
#show_image(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))

cropped = image[50:200, 100:300]
#show_image(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))

# 5. Flip and rotate the image
flipped = cv2.flip(image, 1)  # Horizontal flip
show_image(cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB))

# Rotate 45 degrees around the center
height, width = image.shape[:2]
center = (width // 2, height // 2)
rotation_matrix = cv2.getRotationMatrix2D(center, 45, 1.0)
rotated = cv2.warpAffine(image, rotation_matrix, (width, height))
show_image(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))

# 6. Save the modified image
cv2.imwrite('grayscale_image.png', gray)