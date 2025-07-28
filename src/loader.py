# initialize a chunk loader that uses a generator to load images
import os

def image_generator(folder_path):
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                path = os.path.join(root, filename)
                yield filename, path
