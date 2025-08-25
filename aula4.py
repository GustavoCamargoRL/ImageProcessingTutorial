import cv2
import numpy as np
from matplotlib import pyplot as plt

def show_image(img, cmap=None, title=''):
    plt.figure(figsize=(6,6))
    plt.axis('off')
    if title:
        plt.title(title)
    if len(img.shape) == 2:
        plt.imshow(img, cmap='gray')
    else:
        # OpenCV loads BGR, convert to RGB for display
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

h, w = 240, 360
image = np.zeros((h, w), dtype=np.uint8)
cv2.circle(image, (70, 80), 30, 255, -1)
cv2.circle(image, (180, 80), 30, 255, -1)
cv2.circle(image, (290, 80), 30, 255, -1)
cv2.circle(image, (120, 170), 25, 255, -1)
cv2.circle(image, (250, 170), 25, 255, -1)

rng = np.random.default_rng(42)
noise = rng.choice([0, 255], size=(h, w), p=[0.96, 0.04]).astype(np.uint8)
noisy = cv2.bitwise_or(image, noise)

show_image(noisy, cmap='gray', title='Synthetic image with salt-and-pepper noise')

_, binary = cv2.threshold(noisy, 127, 255, cv2.THRESH_BINARY)
show_image(binary, cmap='gray', title='Binary image (threshold=127)')

kern_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
kern_ellip = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
kern_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

print('Rect kernel:\n', kern_rect)
print('\nEllipse kernel:\n', kern_ellip)
print('\nCross kernel:\n', kern_cross)
