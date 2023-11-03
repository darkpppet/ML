import numpy as np
import matplotlib.pyplot as plt

fruits = np.load('../data/fruits_300.npy')

fig, axs = plt.subplots(1, 3)

axs[0].imshow(fruits[0], cmap='gray_r')
axs[1].imshow(fruits[100], cmap='gray_r')
axs[2].imshow(fruits[200], cmap='gray_r')

plt.show()
