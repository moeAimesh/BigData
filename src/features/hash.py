from PIL import Image
import imagehash

def calc_hash(image):
    return str(imagehash.average_hash(image))