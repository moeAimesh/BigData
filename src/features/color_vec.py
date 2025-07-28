#use Loader.py to load images and extract feature vektores from them
#then save the vektor in database.db 
import numpy as np


def calc_histogram(image):
    hist = image.histogram()
    hist = np.array(hist) / sum(hist)
    return hist