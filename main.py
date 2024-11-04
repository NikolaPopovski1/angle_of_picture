import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import os

# Use an absolute path to the image file
image_path = os.path.abspath('images/primer_-0.25_pi_rad.png')
# image_path = os.path.abspath('images/forest_horison.jpg')
# Load the image
slika = plt.imread(image_path)[:, :, :3]
# Convert to grayscale by averaging the RGB channels
slika = slika.mean(2)




# Plot the image and histogram
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(slika_conv_gaussian, cmap='gray')
plt.axis('off')
plt.subplot(1, 2, 2)
# 16 bins, count is height, width is the difference between two edges
plt.bar(hist_bins[:-1], hist_smeri_x, width=hist_bins[2] - hist_bins[1])
plt.show()