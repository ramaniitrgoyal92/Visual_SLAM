import cv2
import numpy as np
import matplotlib.pyplot as plt

# Initialize the ORB detector
orb = cv2.ORB_create(nfeatures=1000)

# Example usage with an image
image1 = cv2.imread('RamanGoyal.jpg', cv2.IMREAD_GRAYSCALE)

# Detect keypoints and compute descriptors for the first image
keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
 
# Detect keypoints and compute descriptors for the second image
keypoints2, descriptors2 = orb.detectAndCompute(image1, None)

# # Draw keypoints on the image for visualization
# image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0), flags=0)

# plt.imshow(image_with_keypoints)

# # Save or display the image with keypoints
# cv2.imwrite('image_with_keypoints.jpg', image_with_keypoints)


 ###SLAM
 
# Initialize the BFMatcher with default params
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
 
# Match descriptors between frame I and frame I+1
matches = bf.match(descriptors1, descriptors2)
 
# Sort matches by distance (best matches first)
matches = sorted(matches, key=lambda x: x.distance)