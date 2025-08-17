from PIL import Image
import imagehash
import numpy as np


def calc_hash(img):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)  # ← das wandelt NumPy-Bild in PIL.Image
    return str(imagehash.average_hash(img))
