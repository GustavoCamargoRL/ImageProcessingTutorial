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

#print('Rect kernel:\n', kern_rect)
#print('\nEllipse kernel:\n', kern_ellip)
#print('\nCross kernel:\n', kern_cross)

#eroded = cv2.erode(binary, kern_rect, iterations=1)
#dilated = cv2.dilate(binary, kern_rect, iterations=1)

#show_image(eroded, cmap='gray', title='Eroded (3x3 rect)')
#show_image(dilated, cmap='gray', title='Dilated (3x3 rect)')

#opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kern_ellip)
#closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kern_ellip)

#show_image(opening, cmap='gray', title='Opening (5x5 ellipse)')
#show_image(closing, cmap='gray', title='Closing (5x5 ellipse)')

# Apply morphological opening to clean noise
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
show_image(clean, cmap='gray', title='Cleaned binary (opening)')

# Find contours
contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print('Found contours:', len(contours))

# Draw bounding boxes and centroids on a color image
color = cv2.cvtColor(clean, cv2.COLOR_GRAY2BGR)
count = 0
for i, cnt in enumerate(contours, 1):
    area = cv2.contourArea(cnt)
    if area < 200:  # ignore tiny contours
        continue
    count += 1
    x, y, w, h = cv2.boundingRect(cnt)
    cx = x + w//2
    cy = y + h//2
    cv2.rectangle(color, (x,y), (x+w, y+h), (0,255,0), 2)
    cv2.putText(color, f'{count}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.circle(color, (cx, cy), 3, (0,0,255), -1)

show_image(color, title='Detected objects with bounding boxes and centroids')
print('Counted objects (area>200):', count)
