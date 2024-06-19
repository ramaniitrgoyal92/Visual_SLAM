import cv2
import numpy as np
import matplotlib.pyplot as plt

# Create an ORB detector with a specified number of features
orb = cv2.ORB_create(nfeatures=5000)

# Example usage with an image
image = cv2.imread('RamanGoyal.jpg', cv2.IMREAD_GRAYSCALE)
keypoints, descriptors = orb.detectAndCompute(image, None)

# Draw keypoints on the image for visualization
image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0), flags=0)

plt.imshow(image_with_keypoints)

# Save or display the image with keypoints
cv2.imwrite('image_with_keypoints.jpg', image_with_keypoints)
