import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt


slika = plt.imread('image_left.jpg')[:, :, :3]
slika = (slika).mean(2)

f = np.ones((9, 9), dtype=np.float32)
f /= f.sum()

slika_conv_smoothed = ndimage.convolve(slika, f, mode='constant', cval=0.0)

# Define sigma and generate the Gaussian kernel
sigma = 1.5
velikost_jedra = int(3 * sigma)
x = np.arange(-velikost_jedra, velikost_jedra + 1)
X, Y = np.meshgrid(x, x)
jedro_tocke = np.array([X.ravel(), Y.ravel()])  # 2 x P
C = np.eye(2) * 1 / sigma**2
jedro_gauss = np.exp(-1 * (jedro_tocke.T.dot(C) * jedro_tocke.T).sum(1)).reshape(X.shape)

# Normalize the kernel
jedro_gauss /= jedro_gauss.sum()

# Apply the Gaussian kernel to the image using convolution
slika_conv_gaussian = ndimage.convolve(slika, jedro_gauss, mode='constant', cval=0.0)

jedro_sobel_dx = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])
jedro_sobel_dy = jedro_sobel_dx.T

slika_dx = ndimage.convolve(slika_conv_gaussian, jedro_sobel_dx, mode='constant', cval=0.0)
slika_dy = ndimage.convolve(slika_conv_gaussian, jedro_sobel_dy, mode='constant', cval=0.0)
# slika_dx_and_dy = ndimage.convolve(slika_dx, jedro_sobel_dy, mode='constant', cval=0.0)
#"""
slika_rob_mag = (slika_dx**2 + slika_dy**2)**0.5 # samo pitagorov izrek
# slika_rob_smer = np.arctan2(slika_dy, slika_dx)
slika_rob_smer = np.arctan2(slika_dy, slika_dx)
#"""
# Apply Sobel filters to the image
slika_sobel_dx = ndimage.convolve(slika, jedro_sobel_dx, mode='constant', cval=0.0)
slika_sobel_dy = ndimage.convolve(slika, jedro_sobel_dy, mode='constant', cval=0.0)

# Compute the magnitude of the gradient
slika_sobel = np.hypot(slika_sobel_dx, slika_sobel_dy)

# Normalize the result
slika_sobel = (slika_sobel / slika_sobel.max()) * 255  # Scale to 0-255

# Convert to unsigned 8-bit integer type for proper image representation
slika_sobel = slika_sobel.astype(np.uint8)
#"""
# razdelimo histogram na 16 predalckov med vrednostmi -pi, pi
hist_bins = np.linspace(-np.pi, np.pi, 17)
# razdelimo piksle v predalcke po kotu roba, pomnozeno z weightom magnitude ki nam pove kak mocn je rob
hist_smeri_1, _ = np.histogram(slika_sobel, bins=hist_bins, weights=slika_rob_mag)

"""
# Display the original and the blurred image
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title('Original img')
plt.imshow(slika, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Filtered img')
plt.imshow(slika_sobel, cmap='gray')
plt.axis('off')

plt.show()
"""

# Calculate the bin centers
bin_centers = (hist_bins[:-1] + hist_bins[1:]) / 2

# Calculate the weighted average for the x-axis
average_x_rad = np.average(bin_centers, weights=hist_smeri_1)

# Convert the average x-axis value from radians to degrees
average_x_deg = np.degrees(average_x_rad)

# Calculate the average for the y-axis
average_y = np.mean(hist_smeri_1)

print(f"Average value in radians: {average_x_rad}")
print(f"Average value in degrees: {average_x_deg}")

# Find the maximum value on the y-axis (histogram counts)
max_y = np.max(hist_smeri_1)

# Find the index of the maximum value
max_y_index = np.argmax(hist_smeri_1)

# Find the corresponding bin edge (x-axis)
max_x = hist_bins[max_y_index]

# Convert the max x-axis value from radians to degrees
max_x_deg = np.degrees(max_x)

print(f"Maximum value on the x-axis in radians: {max_x}")
print(f"Maximum value on the x-axis in degrees: {max_x_deg}")

# Find the index of the maximum value
max_x_index = np.max

# Find the corresponding bin edge (x-axis)
max_y = hist_bins[max_x_index]

# Convert the max x-axis value from radians to degrees
max_y_deg = np.degrees(max_y)

print(f"Maximum value on the y-axis in radians: {max_y}")
print(f"Maximum value on the y-axis in degrees: {max_y_deg}")

# Plot the image and histogram
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(slika_conv_gaussian, cmap='gray')
plt.axis('off')
plt.subplot(1, 2, 2)
# 16 bins, count is height, width is the difference between two edges
plt.bar(hist_bins[:-1], hist_smeri_1, width=hist_bins[1] - hist_bins[0])
plt.show()
