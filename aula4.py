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