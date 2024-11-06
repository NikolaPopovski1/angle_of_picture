import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import os
import horizont
import matplotlib.patches as patches

# Use an absolute path to the image file
image_path = os.path.abspath('images/image_left_leaning.jpg')
#image_path = os.path.abspath('images/primer_-0.25_pi_rad.png')
# Load the image
slika = plt.imread(image_path)[:, :, :3]

regija1 = slika[1500:1600, 2700:2800, :] # grass, no sky or horizon
regija2 = slika[600:700, 3000:3100, :] # sky and trees & horizon
regija2 = ndimage.rotate(regija2, 50) # rotate the region
regija3 = slika[1265:1365, 1690:1790, :] # grass with shadow acting as horizon
regija4 = slika[1070:1170, 1360:1460, :] #trees, a cloud and sky in a form of a horizon
regija_flip = slika[210:310, 275:375, :] #trees, a cloud and sky in a form of a horizon on another location
regija5 = regija_flip[::-1, :, :] #flipping region 5 upside down


#print(f"Kot je {horizont.orientacija_horizonta(slika)} radianov.")
#"""
print(f"Kot za regijo 1 je {horizont.orientacija_horizonta(regija1)} radianov.")
print(f"Kot za regijo 2 je {horizont.orientacija_horizonta(regija2)} radianov.")
print(f"Kot za regijo 3 je {horizont.orientacija_horizonta(regija3)} radianov.")
print(f"Kot za regijo 4 je {horizont.orientacija_horizonta(regija4)} radianov.")
print(f"Kot za regijo 5 je {horizont.orientacija_horizonta(regija5)} radianov.")
#"""
#"""
plt.figure(figsize=(15, 5))

plt.subplot(1, 5, 1)
plt.title('Regija 1')
plt.axis('off')
plt.imshow(regija1)

plt.subplot(1, 5, 2)
plt.title('Regija 2')
plt.axis('off')
plt.imshow(regija2)

plt.subplot(1, 5, 3)
plt.title('Regija 3')
plt.axis('off')
plt.imshow(regija3)

plt.subplot(1, 5, 4)
plt.title('Regija 4')
plt.axis('off')
plt.imshow(regija4)

plt.subplot(1, 5, 5)
plt.title('Regija 5')
plt.axis('off')
plt.imshow(regija5)

plt.show()
#"""
"""
fig, ax = plt.subplots(1, figsize=(15, 10))
ax.imshow(slika)
regions = [
    (1500, 1600, 2700, 2800, 'Regija 1'),
    (600, 700, 3000, 3100, 'Regija 2'),
    (1265, 1365, 1690, 1790, 'Regija 3'),
    (1070, 1170, 1360, 1460, 'Regija 4'),
    (210, 310, 275, 375, 'Regija 5')
]

for (y1, y2, x1, x2, title) in regions:
    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    ax.text((x1 + x2) / 2, y1-10, title, color='r', fontsize=12, weight='normal', ha='center')

ax.axis('off')
plt.show()
"""

# Run the tests
