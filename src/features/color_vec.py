import numpy as np
import cv2

def calc_histogram(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hist = []
    for i in range(3):  # R, G, B
        h = cv2.calcHist([img_rgb], [i], None, [32], [0, 256])
        h = cv2.normalize(h, h).flatten()
        hist.append(h)
    return np.concatenate(hist)