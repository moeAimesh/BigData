#use Loader.py to load images and extract feature vektores from them
#then save the vektor in database.db 
import numpy as np


def calc_histogram(image):
    arr = np.array(image.convert("RGB")).reshape(-1, 3)
    hist_r, _ = np.histogram(arr[:, 0], bins=32, range=(0, 256), density=True)
    hist_g, _ = np.histogram(arr[:, 1], bins=32, range=(0, 256), density=True)
    hist_b, _ = np.histogram(arr[:, 2], bins=32, range=(0, 256), density=True)
    return np.concatenate([hist_r, hist_g, hist_b])