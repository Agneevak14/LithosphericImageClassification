import matplotlib.pyplot as plt
import numpy as np

# Example RGB values (0–255)
R, G, B = 0, 115, 235

# Normalize to 0–1 range
color = np.array([[ [R/255, G/255, B/255] ]])

plt.imshow(color)
plt.axis('off')
plt.show()
