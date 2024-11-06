import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import os
import horizont

# Use an absolute path to the image file
#image_path = os.path.abspath('images/image_left_leaning.jpg')
image_path = os.path.abspath('images/primer_0_pi_rad.png')
# Load the image
slika = plt.imread(image_path)[:, :, :3]

#regija1 = slika[500:600, 2700:2800, :]
#regija2 = slika[600:700, 3000:3100, :]

# Pass regija2 to the function
print(f"Kot je {horizont.orientacija_horizonta(slika)} radianov.")
"""
plt.figure()
plt.title('izbrana regija zanimanja')
plt.axis('off')
plt.imshow(slika)
plt.show()
"""