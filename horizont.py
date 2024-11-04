import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt

def orientacija_horizonta(slika: np.ndarray) -> float:
    # Apply gaussian filter from ndimage library
    slika_conv_gaussian = ndimage.gaussian_filter(slika, sigma=10,radius=51,mode="mirror")


    jedro_sobel_dx = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    jedro_sobel_dy = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ])

    slika_dx = ndimage.convolve(slika_conv_gaussian, jedro_sobel_dx)
    slika_dy = ndimage.convolve(slika_conv_gaussian, jedro_sobel_dy)
    # slika_dx_and_dy = ndimage.convolve(slika_dx, jedro_sobel_dy, mode='constant', cval=0.0)
    #"""



    slika_rob_mag = (slika_dx**2 + slika_dy**2)**0.5 # samo pitagorov izrek
    # slika_rob_smer = np.arctan2(slika_dy, slika_dx)
    slika_rob_smer = np.arctan2(slika_dx, slika_dy)
    """
    # Apply Sobel filters to the image
    slika_sobel_dx = ndimage.convolve(slika_conv_gaussian, jedro_sobel_dx)
    slika_sobel_dy = ndimage.convolve(slika_conv_gaussian, jedro_sobel_dy)
    # Compute the magnitude of the gradient
    slika_sobel = np.hypot(slika_sobel_dx, slika_sobel_dy)
    # Normalize the result
    slika_sobel = (slika_sobel / slika_sobel.max()) * 255  # Scale to 0-255
    # Convert to unsigned 8-bit integer type for proper image representation
    slika_sobel = slika_sobel.astype(np.uint8)
    """
    
    # razdelimo histogram na 16 predalckov med vrednostmi -pi, pi
    hist_bins = np.linspace(-np.pi, np.pi, 100)
    # razdelimo piksle v predalcke po kotu roba, pomnozeno z weightom magnitude ki nam pove kak mocn je rob
    # hist_smeri_x, hist_smeri_y = np.histogram(slika_sobel, bins=hist_bins, weights=slika_rob_mag)
    # Tried both methods for edge detection but the results are not as expected
    hist_smeri_x, _ = np.histogram(slika_rob_smer, bins=hist_bins, weights=slika_rob_mag)

    # Find the index of the maximum value
    max_y_index = np.argmax(hist_smeri_x)
    # Find the corresponding bin edge (x-axis)
    max_x = hist_bins[max_y_index]
    # Find the value next to it
    max_x_plus_one = hist_bins[max_y_index + 1]
    
    result = (max_x_plus_one + max_x) * 0.5
    
    """
    print(f"Maximum value on the x-axis (bin edge in rad): {max_x}")
    # Find the second largest value in the histogram counts
    sorted_hist_smeri_x = np.sort(hist_smeri_x)[::-1]  # Sort in descending order
    second_largest_y = sorted_hist_smeri_x[1]  # Second largest value
    # Find the corresponding bin edge for the second largest value
    second_largest_y_index = np.where(hist_smeri_x == second_largest_y)[0][0]
    second_largest_x = hist_bins[second_largest_y_index]
    print(f"Second largest value on the x-axis (bin edge in rad): {second_largest_x}")
    print(f"")
    print(f"Maximum and value next to it combined and devided by 2:")
    print(f"Angle in rad: {result}")
    print(f"Agle in degrees: {result * 180 / np.pi}")
    #max_y = x[np.where(hist_smeri_y == hist_smeri_y.max())]
    #print(f"Maximum: {hist_bins[max_y]}")


    # Plot the image and histogram
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(slika, cmap='gray')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    # 16 bins, count is height, width is the difference between two edges
    plt.bar(hist_bins[:-1], hist_smeri_x, width=hist_bins[2] - hist_bins[1])
    plt.show()
    """

    return result