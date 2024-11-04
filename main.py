import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import os
import horizont

# Use an absolute path to the image file
# image_path = os.path.abspath('images/primer_-0.25_pi_rad.png')
image_path = os.path.abspath('images/image_left_leaning.jpg')
# Load the image
slika = plt.imread(image_path)[:, :, :3]

regija = slika[:, :, :]

# Convert to grayscale by averaging the RGB channels
slika = regija.mean(2)

print(f"Kot je {horizont.orientacija_horizonta(slika)} radianov.")

plt.figure()
plt.title('izbrana regija zanimanja')
plt.axis('off')
plt.imshow(slika)
plt.show()
